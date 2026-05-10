use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub theta: f32,     // azimuth angle (horizontal)
    pub phi: f32,       // polar angle from +Y down
    pub distance: f32,
    pub fov_y_deg: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            theta: std::f32::consts::FRAC_PI_4,
            phi: std::f32::consts::FRAC_PI_4,
            distance,
            fov_y_deg: 45.0,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.phi.sin() * self.theta.cos(),
                self.distance * self.phi.cos(),
                self.distance * self.phi.sin() * self.theta.sin(),
            )
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y_deg.to_radians(), aspect, 0.01, 1000.0);
        proj * view
    }

    /// Drag: delta_x rotates around Y, delta_y tilts.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.theta -= dx * 0.008;
        self.phi = (self.phi - dy * 0.008).clamp(0.02, std::f32::consts::PI - 0.02);
    }

    /// Positive delta = zoom in.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.8).max(0.5);
    }
}
