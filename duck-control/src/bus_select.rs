//! Which bus the daemon opens — chosen by `[bus] protocol`, not at compile time.
//!
//! `robotd` reaches past [`RobotIo`] for a handful of startup operations — the ping census,
//! the register check, the motor-swap path, the `init` ramp — that are inherent methods of a
//! bus backend rather than part of the trait, and [`crate::safety::Safety`] is generic over its
//! io. A `Box<dyn RobotIo>` would serve neither, so this is an enum with the same surface as
//! one backend, and every method is a two-arm `match`.
//!
//! The default is the Dynamixel bus: a parameter file that says nothing about a protocol
//! behaves exactly as it did before this module existed.

use std::time::Duration;

use crate::bus::DynamixelIo;
use crate::bus_feetech::{FeetechConfig, FeetechIo};
use crate::io::{ImuStale, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::NUM_JOINTS;

/// The wire the servos speak.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum BusProtocol {
    /// Dynamixel Protocol 2.0, XL330 servos, the `imu_to_dxl` board at ID 200.
    #[default]
    Dynamixel2,
    /// Feetech STS/SMS framing, HD-1910 servos.
    FeetechSts(FeetechConfig),
}

impl BusProtocol {
    /// The spelling used in `robotd.toml`.
    pub const DYNAMIXEL2: &'static str = "dynamixel2";
    pub const FEETECH_STS: &'static str = "feetech-sts";

    pub fn name(&self) -> &'static str {
        match self {
            Self::Dynamixel2 => Self::DYNAMIXEL2,
            Self::FeetechSts(_) => Self::FEETECH_STS,
        }
    }
}

/// The two backends are not the same size: the Feetech one carries an IMU decoder and a
/// stale-block tracker, which puts it far enough past the Dynamixel one that clippy suggests
/// boxing it. There is one of these per process and it is never copied, so the size costs
/// nothing anywhere — while the box would buy that nothing by putting an indirection in front
/// of every method call the control loop makes.
#[allow(clippy::large_enum_variant)]
pub enum AnyBus {
    Dynamixel(DynamixelIo),
    Feetech(FeetechIo),
}

/// One arm per backend, same expression in each.
macro_rules! each {
    ($self:expr, $io:ident => $body:expr) => {
        match $self {
            AnyBus::Dynamixel($io) => $body,
            AnyBus::Feetech($io) => $body,
        }
    };
}

impl AnyBus {
    pub fn open(port: &str, protocol: &BusProtocol) -> Result<Self> {
        Ok(match protocol {
            BusProtocol::Dynamixel2 => Self::Dynamixel(DynamixelIo::open(port)?),
            BusProtocol::FeetechSts(config) => {
                Self::Feetech(FeetechIo::open(port, config.clone())?)
            }
        })
    }

    pub fn protocol_name(&self) -> &'static str {
        match self {
            Self::Dynamixel(_) => BusProtocol::DYNAMIXEL2,
            Self::Feetech(_) => BusProtocol::FEETECH_STS,
        }
    }

    pub fn check_registers(&mut self) -> Result<usize> {
        each!(self, io => io.check_registers())
    }

    pub fn missing_servos(&mut self) -> Result<Vec<u8>> {
        each!(self, io => io.missing_servos())
    }

    pub fn adopt_replacement(&mut self, id: u8) -> Result<bool> {
        each!(self, io => io.adopt_replacement(id))
    }

    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        each!(self, io => io.present_positions())
    }

    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        each!(self, io => io.interpolate_to(target, duration, step))
    }
}

impl RobotIo for AnyBus {
    fn read(&mut self) -> Result<Sensors> {
        each!(self, io => io.read())
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        each!(self, io => io.write(targets))
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        each!(self, io => io.set_gain(kp))
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        each!(self, io => RobotIo::set_torque(io, on))
    }

    fn reboot(&mut self, id: u8) -> Result<()> {
        each!(self, io => io.reboot(id))
    }

    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        each!(self, io => io.slow_sensors())
    }

    fn imu_stale(&self) -> ImuStale {
        each!(self, io => io.imu_stale())
    }

    fn imu_ready(&self) -> bool {
        each!(self, io => io.imu_ready())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names are what a person types into `robotd.toml`; the default has to stay the bus
    /// every existing robot has.
    #[test]
    fn the_default_protocol_is_the_dynamixel_bus() {
        assert_eq!(BusProtocol::default(), BusProtocol::Dynamixel2);
        assert_eq!(BusProtocol::default().name(), "dynamixel2");
        assert_eq!(
            BusProtocol::FeetechSts(FeetechConfig::default()).name(),
            "feetech-sts"
        );
    }
}
