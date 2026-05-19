#requires -version 5.1
<#
build-pack.ps1 - generate a VJ pack from one render mode x 16 crystals.

Per clip: render PNG sequence via `crystal-viz --render`, encode HAP-Q .mov
(Resolume/VDMX deliverable) + H.264 .mp4 (preview). Then assemble a 4x4 contact
sheet and a sizzle reel, plus metadata.json.

Examples:
  powershell scripts/build-pack.ps1
  powershell scripts/build-pack.ps1 -Bpm 128 -Bars 8 -Resolution 1920x1080
  powershell scripts/build-pack.ps1 -SmokeTest        # 2 clips at 640x360 to validate
#>

[CmdletBinding()]
param(
    [string]$OutDir     = "packs/nodal_120bpm",
    [string]$BasePreset = "presets/nodal_loop.preset.json",
    [int]   $Bpm        = 120,
    [int]   $Bars       = 8,
    [int]   $Fps        = 60,
    [string]$Resolution = "1920x1080",
    [string]$ScratchDir = "",     # PNG scratch root; default = $env:TEMP
    [switch]$SkipMov,
    [switch]$KeepFrames,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$script_root = Split-Path -Parent $PSCommandPath
$repo_root   = Split-Path -Parent $script_root
Set-Location $repo_root

$binary = Join-Path $repo_root "target/release/crystal-viz.exe"
if (-not (Test-Path $binary)) {
    throw "binary not found at $binary - run: cargo build --release --bin crystal-viz"
}
if (-not (Test-Path $BasePreset)) {
    throw "base preset not found: $BasePreset"
}
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpeg) { throw "ffmpeg not on PATH" }

# PowerShell 5.1's `Set-Content -Encoding utf8` writes a BOM that serde_json
# rejects. Use this helper to write BOM-less UTF-8.
$utf8_no_bom = New-Object System.Text.UTF8Encoding($false)
function Write-Utf8NoBom([string]$Path, [string]$Content) {
    [System.IO.File]::WriteAllText($Path, $Content, $utf8_no_bom)
}

# Curated set spanning the seven crystal systems so NODAL line patterns
# vary visibly across clips (cubic = blocky, hex = 6-fold, etc.).
$crystals = @(
    @{ name = "BCC Iron";        slug = "bcc_iron" },
    @{ name = "FCC Gold";        slug = "fcc_gold" },
    @{ name = "Diamond";         slug = "diamond" },
    @{ name = "NaCl Rock salt";  slug = "nacl" },
    @{ name = "Pyrite";          slug = "pyrite" },
    @{ name = "Perovskite";      slug = "perovskite" },
    @{ name = "Zincblende";      slug = "zincblende" },
    @{ name = "Graphite";        slug = "graphite" },
    @{ name = "HCP Magnesium";   slug = "hcp_mg" },
    @{ name = "Beryl";           slug = "beryl" },
    @{ name = "Rutile";          slug = "rutile" },
    @{ name = "Zircon";          slug = "zircon" },
    @{ name = "alpha-Quartz";    slug = "alpha_quartz" },
    @{ name = "Calcite";         slug = "calcite" },
    @{ name = "Aragonite";       slug = "aragonite" },
    @{ name = "Forsterite";      slug = "forsterite" }
)

if ($SmokeTest) {
    $crystals   = $crystals[0..1]
    $Resolution = "640x360"
    if ($OutDir -eq "packs/nodal_120bpm") { $OutDir = "packs/_smoke" }
    Write-Host "[smoke] 2 clips @ $Resolution -> $OutDir" -ForegroundColor Yellow
}

$clip_count    = $crystals.Count
$duration_s    = [double]$Bars * 4.0 * 60.0 / [double]$Bpm
$frames_total  = [int]($Fps * $duration_s)
$mid_frame     = [int]([Math]::Floor($frames_total / 2.0))
$clip_dir      = Join-Path $OutDir "clips"
$scratch_base  = if ($ScratchDir) { $ScratchDir } else { [IO.Path]::GetTempPath() }
$tmp_root      = Join-Path $scratch_base ("cd_pack_" + [Guid]::NewGuid().ToString("N").Substring(0,8))

New-Item -Force -ItemType Directory -Path $clip_dir | Out-Null
New-Item -Force -ItemType Directory -Path $tmp_root | Out-Null

Write-Host ""
Write-Host "Pack       : $OutDir" -ForegroundColor Cyan
Write-Host "Base preset: $BasePreset"
Write-Host "Tempo      : $Bpm BPM x $Bars bars = $([Math]::Round($duration_s,3))s ($frames_total frames @ $Fps fps)"
Write-Host "Resolution : $Resolution"
$codec_line = "H.264 mp4"
if (-not $SkipMov) { $codec_line += " + HAP-Q mov" }
Write-Host "Codecs     : $codec_line"
Write-Host "Temp scratch: $tmp_root"
Write-Host ""

$base_json = Get-Content $BasePreset -Raw

$manifest = @()
$failed   = @()
$total_start = Get-Date

for ($i = 0; $i -lt $clip_count; $i++) {
    $c        = $crystals[$i]
    $hue      = [Math]::Round($i / [double]$clip_count, 6)
    $slug     = "{0:D2}_{1}" -f ($i + 1), $c.slug
    $idx_str  = "[{0:D2}/{1:D2}]" -f ($i + 1), $clip_count
    Write-Host "$idx_str $($c.name)  hue+=$hue" -ForegroundColor Cyan

    # Per-clip preset: rotate every step's color_shift by $hue.
    $clip_preset = $base_json | ConvertFrom-Json
    $clip_preset.name = "Nodal - $($c.name)"
    foreach ($s in $clip_preset.steps) {
        $cs = ($s.params.color_shift + $hue) % 1.0
        $s.params.color_shift = [Math]::Round($cs, 6)
    }
    $preset_path = Join-Path $tmp_root ($slug + ".preset.json")
    Write-Utf8NoBom $preset_path ($clip_preset | ConvertTo-Json -Depth 12)

    $frame_dir = Join-Path $tmp_root ("frames_" + $slug)
    New-Item -Force -ItemType Directory -Path $frame_dir | Out-Null

    try {
        $clip_start = Get-Date
        & $binary `
            --render   $preset_path `
            --crystal  $c.name `
            --bpm      $Bpm `
            --bars     $Bars `
            --fps      $Fps `
            --res      $Resolution `
            --out      $frame_dir | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "crystal-viz --render exited $LASTEXITCODE" }

        $frame_pattern = Join-Path $frame_dir "frame_%06d.png"

        # H.264 preview
        $mp4 = Join-Path $clip_dir ($slug + ".mp4")
        & ffmpeg -y -loglevel error `
            -framerate $Fps -i $frame_pattern `
            -c:v libx264 -pix_fmt yuv420p -crf 20 -preset fast -movflags +faststart `
            $mp4
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg mp4 exited $LASTEXITCODE" }

        # HAP-Q .mov - actual VJ deliverable
        $mov = $null
        if (-not $SkipMov) {
            $mov = Join-Path $clip_dir ($slug + ".mov")
            & ffmpeg -y -loglevel error `
                -framerate $Fps -i $frame_pattern `
                -c:v hap -format hap_q `
                $mov
            if ($LASTEXITCODE -ne 0) { throw "ffmpeg hap exited $LASTEXITCODE" }
        }

        # Stash mid-frame for the contact sheet (before we delete PNGs).
        $thumb_src = Join-Path $frame_dir ("frame_{0:D6}.png" -f $mid_frame)
        $thumb_dst = Join-Path $tmp_root ("thumb_{0:D2}.png" -f ($i + 1))
        if (Test-Path $thumb_src) { Copy-Item $thumb_src -Destination $thumb_dst }

        $clip_elapsed = (Get-Date) - $clip_start
        $mp4_size = (Get-Item $mp4).Length
        $mov_size = if ($mov -and (Test-Path $mov)) { (Get-Item $mov).Length } else { 0 }
        $size_line = "mp4 {0:F1} MB" -f ($mp4_size/1MB)
        if ($mov_size) { $size_line += ", mov {0:F1} MB" -f ($mov_size/1MB) }
        Write-Host ("        done in {0:F1}s - {1}" -f $clip_elapsed.TotalSeconds, $size_line)

        $manifest += [ordered]@{
            index               = $i + 1
            slug                = $slug
            crystal             = $c.name
            mode                = "NODAL"
            bpm                 = $Bpm
            bars                = $Bars
            duration_s          = [Math]::Round($duration_s, 6)
            fps                 = $Fps
            resolution          = $Resolution
            color_shift_offset  = $hue
            preview_mp4         = "clips/$slug.mp4"
            hap_q_mov           = if ($mov) { "clips/$slug.mov" } else { $null }
        }
    } catch {
        Write-Host ("        FAILED: {0}" -f $_) -ForegroundColor Red
        $failed += @{ slug = $slug; crystal = $c.name; error = "$_" }
    } finally {
        if (-not $KeepFrames -and (Test-Path $frame_dir)) {
            Remove-Item -Recurse -Force $frame_dir
        }
    }
}

# Contact sheet (NxM grid of mid-clip thumbs)
$sheet_path = Join-Path $OutDir "contact_sheet.png"
$grid_w = [int][Math]::Ceiling([Math]::Sqrt($clip_count))
$grid_h = [int][Math]::Ceiling($clip_count / [double]$grid_w)
$thumb_pattern = Join-Path $tmp_root "thumb_%02d.png"
$thumbs_present = (Get-ChildItem -Path $tmp_root -Filter "thumb_*.png" -ErrorAction SilentlyContinue).Count
if ($thumbs_present -gt 0) {
    & ffmpeg -y -loglevel error `
        -framerate 1 -start_number 1 -i $thumb_pattern `
        -vf "scale=480:-1,tile=${grid_w}x${grid_h}" `
        $sheet_path
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Contact sheet: $sheet_path"
    } else {
        Write-Host "contact sheet ffmpeg exited $LASTEXITCODE" -ForegroundColor Yellow
    }
}

# Sizzle reel - short slice from each clip, concatenated
$sizzle_seconds = [Math]::Min(4.0, $duration_s / 2.0)
$reel_path   = Join-Path $OutDir "preview.mp4"
$cuts_list   = Join-Path $tmp_root "concat.txt"
$reel_built  = $false
if ((Get-ChildItem -Path $clip_dir -Filter "*.mp4" -ErrorAction SilentlyContinue).Count -gt 0) {
    Remove-Item $cuts_list -ErrorAction SilentlyContinue
    foreach ($entry in $manifest) {
        $src = Join-Path $clip_dir ($entry.slug + ".mp4")
        $cut = Join-Path $tmp_root ("cut_{0:D2}.mp4" -f $entry.index)
        & ffmpeg -y -loglevel error `
            -ss 2 -t $sizzle_seconds -i $src `
            -c:v libx264 -pix_fmt yuv420p -crf 22 -preset fast `
            $cut
        if ($LASTEXITCODE -eq 0 -and (Test-Path $cut)) {
            Add-Content -Path $cuts_list -Value ("file '{0}'" -f ($cut -replace "\\", "/"))
        }
    }
    if (Test-Path $cuts_list) {
        & ffmpeg -y -loglevel error -f concat -safe 0 -i $cuts_list `
            -c:v libx264 -pix_fmt yuv420p -crf 20 -preset fast -movflags +faststart `
            $reel_path
        if ($LASTEXITCODE -eq 0) { $reel_built = $true; Write-Host "Sizzle reel  : $reel_path" }
    }
}

# Metadata
$meta_path = Join-Path $OutDir "metadata.json"
$contact_sheet_rel = if (Test-Path $sheet_path) { "contact_sheet.png" } else { $null }
$preview_rel       = if ($reel_built) { "preview.mp4" } else { $null }
$meta = [ordered]@{
    pack_name     = "crystaldive NODAL - $clip_count crystals"
    mode          = "NODAL"
    generated_at  = (Get-Date).ToString("o")
    bpm           = $Bpm
    bars          = $Bars
    duration_s    = [Math]::Round($duration_s, 6)
    fps           = $Fps
    resolution    = $Resolution
    clip_count    = $manifest.Count
    failed_count  = $failed.Count
    failed        = $failed
    contact_sheet = $contact_sheet_rel
    preview       = $preview_rel
    clips         = $manifest
}
Write-Utf8NoBom $meta_path ($meta | ConvertTo-Json -Depth 12)

if (-not $KeepFrames) { Remove-Item -Recurse -Force $tmp_root -ErrorAction SilentlyContinue }

$elapsed = (Get-Date) - $total_start
Write-Host ""
Write-Host "------------------------------------------------------" -ForegroundColor Green
Write-Host "Pack ready: $OutDir" -ForegroundColor Green
Write-Host "  Clips ok / failed : $($manifest.Count) / $($failed.Count)"
Write-Host ("  Total time        : {0:F1} min" -f $elapsed.TotalMinutes)
Write-Host "  Metadata          : $meta_path"
