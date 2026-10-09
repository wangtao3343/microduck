//! The Feetech bus — STS/SMS-protocol servos (HD-1910) in place of the XL330s.
//!
//! The same shape as [`crate::bus`]: one `sync_read` per tick, one `sync_write` of goal
//! positions, and the same startup census, register check and motor-swap path. What differs is
//! the wire — Feetech's own framing (`FF FF id len instr … ~sum`), which `rustypot` carries as
//! its "protocol v1" together with the STS register table — and four facts about the servo
//! that change what the code around the wire has to do:
//!
//! - **One block holds everything.** Position, speed, load, voltage, temperature, status and
//!   current sit contiguously at 56–70. The tick's read already carries the supply voltage and
//!   the case temperature, so [`RobotIo::slow_sensors`] costs no transaction of its own here.
//! - **The gains live in EEPROM.** P/D/I are at 21–23, inside the EEPROM area. The servo's
//!   write lock (register 55) ships *closed*, and a write to an EEPROM address while it is
//!   closed is accepted but not persisted — which is exactly the RAM semantics the XL330's
//!   gains have, and what [`RobotIo::set_gain`] relies on. Nothing here opens the lock for a
//!   gain write, so a limp fall does not cost an EEPROM cycle.
//! - **The sign of a joint is not a given.** On the XL330 robot the horn orientation fixes the
//!   positive direction and it matches the model. With different servos in remodelled parts
//!   that has to be measured per joint, so it is a parameter: [`FeetechConfig::directions`].
//! - **The IMU answers in the same `sync_read`.** The `imu_to_dxl` board's firmware speaks the
//!   same Feetech framing as the servos and maps its sample onto the same register window
//!   ([`READ_ADDR`] · [`READ_LEN`]), so one transaction still carries joints and orientation
//!   together — the same "one transaction" [`crate::bus`] makes. It is asked **last** in the
//!   list, not first: the board fires its reply a fixed delay after the instruction while a
//!   servo's own response delay is shorter, so with the board at the head the two replies
//!   collide and both arrive corrupt. Measured on the bench — `tools/下游改动.md`, 改动 C.
//!
//! Register addresses and semantics are from Feetech's *磁编码 STS 内存表* v1.1 (2026-08-27) and
//! *SCS 通信协议* v1.0 (2026-06-09); `rustypot`'s `sts3215` table agrees with them.

use std::f64::consts::PI;
use std::time::Duration;

use rustypot::servo::feetech::sts3215::Sts3215Controller;

use crate::bus::{STALE_RUN_WARN, StaleImuTracker};
use crate::imu::{IMU_BLOCK_LEN, SflpDecoder};
use crate::io::{ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{
    BAUD_RATE, FACTORY_ID, FEETECH_EXPECTED_MODE, IMU_DXL_ID, JOINT_IDS, JOINT_NAMES, NUM_JOINTS,
};

/// What `rustypot` returns. Its own alias is private, so it is spelled out here.
type RpResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Start of the feedback block: `present_position` (56, 2) · `present_speed` (58, 2) ·
/// `present_load` (60, 2) · `present_voltage` (62) · `present_temperature` (63) · async-write
/// flag (64) · `status` (65) · `moving` (66) · goal readback (67, 2) · `present_current` (69, 2).
const READ_ADDR: u8 = 56;
const READ_LEN: u8 = 15;

/// `goal_position`, two bytes, little-endian. Written raw rather than through rustypot's
/// `AnglePosition` so the per-joint direction and the 0–4095 clamp are applied in one place.
const GOAL_POSITION_ADDR: u8 = 42;

/// Version block: firmware major/minor, the endianness flag, servo major/minor.
const VERSION_ADDR: u8 = 0;
const VERSION_LEN: u8 = 5;

/// Encoder steps per revolution, and the register value that means 0 rad.
const STEPS_PER_REV: f64 = 4096.0;
const RAW_MAX: i32 = 4095;

/// `present_current` counts 6.5 mA each (STS table, register 69).
const MILLIAMPS_PER_COUNT: f64 = 6.5;

/// `present_voltage` counts 0.1 V each.
const VOLTS_PER_COUNT: f64 = 0.1;

/// Same bound as the Dynamixel bus, for the same reason: a missing device must cost a bounded
/// hiccup, not the serial driver's default.
const READ_TIMEOUT: Duration = Duration::from_millis(30);

/// The protocol document gives "about 800 ms" for the REBOOT instruction (0x08). The XL330
/// path waits 500 ms; that would ping a Feetech servo while it is still down and fail an
/// adoption that worked.
const REBOOT_SETTLE: Duration = Duration::from_millis(1100);

/// Pause after each EEPROM write, as on the other bus.
const EEPROM_SETTLE: Duration = Duration::from_millis(20);

/// `phase` (register 18) bit 2: speed registers count 0.0146 rpm — one encoder step per second —
/// when set, and 0.732 rpm — fifty steps per second — when clear.
const PHASE_SPEED_UNIT_FINE: u8 = 1 << 2;

/// What this backend needs to know that the bus cannot tell it.
#[derive(Debug, Clone, PartialEq)]
pub struct FeetechConfig {
    /// Feetech `P` per unit of the XL330 `position_p_gain` the rest of the daemon thinks in.
    ///
    /// The control loop's gains (200 running, 160 standing, 50 limp) are XL330 register
    /// values on a 0–16383 scale; the STS `P` is one byte. The factory HD-1910 value is 32, and
    /// 32 / 200 = 0.16 is the starting point — it makes "running gain" mean "factory stiffness".
    /// Calibrate against a step response on the bench; see `software/飞特适配架构.md` §3.3.
    pub p_scale: f64,
    /// +1 or −1 per joint, indexed as [`JOINT_NAMES`]: whether the servo's increasing count is
    /// the model's positive rotation. Measured on the robot, not derived.
    pub directions: [f64; NUM_JOINTS],
}

impl Default for FeetechConfig {
    fn default() -> Self {
        Self {
            p_scale: 0.16,
            directions: [1.0; NUM_JOINTS],
        }
    }
}

impl FeetechConfig {
    /// Build from the parameter file's plain values, refusing anything that is not ±1 — a
    /// direction of 0 would pin a joint at its zero, and 2 would double every angle.
    pub fn new(p_scale: f64, directions: &[i8]) -> std::result::Result<Self, String> {
        if directions.len() != NUM_JOINTS {
            return Err(format!(
                "bus.feetech.directions has {} entries, expected {NUM_JOINTS}",
                directions.len()
            ));
        }
        if !(p_scale.is_finite() && p_scale > 0.0) {
            return Err(format!(
                "bus.feetech.p_scale must be positive, got {p_scale}"
            ));
        }
        let mut out = [1.0; NUM_JOINTS];
        for (i, &d) in directions.iter().enumerate() {
            out[i] = match d {
                1 => 1.0,
                -1 => -1.0,
                other => {
                    return Err(format!(
                        "bus.feetech.directions[{i}] ({}) must be 1 or -1, got {other}",
                        JOINT_NAMES[i]
                    ));
                }
            };
        }
        Ok(Self {
            p_scale,
            directions: out,
        })
    }
}

pub struct FeetechIo {
    controller: Sts3215Controller,
    config: FeetechConfig,
    /// The `sync_read` list: the joints in [`JOINT_IDS`] order, then the IMU board **last**.
    /// Held rather than rebuilt each tick, and the order is load-bearing — [`read_ids`] says
    /// why.
    ids: Vec<u8>,
    /// Encoder steps per second for one count of a speed register — 1 or 50, from `phase`.
    /// Read at [`Self::check_registers`]; 1 until then, which is what the servos ship with.
    speed_steps_per_count: f64,
    /// Voltage and temperatures from the last tick's block, so `slow_sensors` is free.
    last_slow: Option<SlowSensors>,
    imu: SflpDecoder,
    /// Blocks identical to their predecessor: the board answering without refreshing. The same
    /// fault [`crate::bus::DynamixelIo`] counts, for the same reason — a frozen orientation is
    /// invisible unless somebody counts it, and on that bus it was seen to happen.
    stale_imu: StaleImuTracker,
}

impl FeetechIo {
    pub fn open(port: &str, config: FeetechConfig) -> Result<Self> {
        Ok(Self {
            controller: open_controller(port, BAUD_RATE)?,
            config,
            ids: read_ids(),
            speed_steps_per_count: 1.0,
            last_slow: None,
            imu: SflpDecoder::default(),
            stale_imu: StaleImuTracker::default(),
        })
    }

    /// Assert the registers the control loop depends on, and correct the one that is safe to.
    ///
    /// Returns how many were corrected. Only the baud rate is ever written: a servo at another
    /// speed would not have answered the census, so in practice this finds nothing, and it is
    /// here for the replacement path. The rest is reported and left alone on purpose —
    ///
    /// - `mode`: the HD-1910 ships in mode 4, "pure position PD", which is what the policy is
    ///   to be trained against. A servo in another mode still answers and still moves, just
    ///   differently; rewriting fifteen EEPROMs on a guess is worse than saying so.
    /// - `response_status_level`: must be 1. At 0 the servo acknowledges nothing but reads, and
    ///   every `rustypot` write then costs a 30 ms timeout and reports failure. That cannot be
    ///   worked around from here, so it is an error rather than a warning.
    ///
    /// Also reads `phase` once, which decides the unit of every speed register.
    pub fn check_registers(&mut self) -> Result<usize> {
        let mut fixed = 0;
        for &id in &JOINT_IDS {
            fixed += self.check_registers_of(id)?;
        }
        self.read_speed_unit()?;
        Ok(fixed)
    }

    fn check_registers_of(&mut self, id: u8) -> Result<usize> {
        let version = self
            .controller
            .read_raw_data(id, VERSION_ADDR, VERSION_LEN)
            .map_err(|e| IoError::Bus(format!("read version block on {id}: {e}")))?;
        if version.len() != VERSION_LEN as usize {
            return Err(IoError::ShortRead {
                what: "version block",
                expected: VERSION_LEN as usize,
                got: version.len(),
            });
        }
        // Register 2 is the endianness flag; 0 is little-endian, which every decode below
        // assumes. The potentiometer SCS family is big-endian and shares the instruction set.
        if version[2] != 0 {
            return Err(IoError::Bus(format!(
                "servo {id} reports big-endian registers (END = {}); this backend is for the \
                 magnetic-encoder STS/SMS family",
                version[2]
            )));
        }

        let level = first(
            self.controller.read_response_status_level(id),
            "response_status_level",
            id,
        )?;
        if level != 1 {
            return Err(IoError::Bus(format!(
                "servo {id} has response_status_level = {level}; it must be 1 or no write is \
                 ever acknowledged — set it with the Feetech debug tool"
            )));
        }

        let mode = first(self.controller.read_mode(id), "mode", id)?;
        if mode != FEETECH_EXPECTED_MODE {
            tracing::warn!(
                id,
                mode,
                want = FEETECH_EXPECTED_MODE,
                firmware = %format!("{}.{}", version[0], version[1]),
                "servo is not in the expected operating mode; left as it is"
            );
        }

        let mut fixed = 0;
        let baud = first(self.controller.read_baudrate(id), "baudrate", id)?;
        if baud != 0 {
            tracing::warn!(
                id,
                got = baud,
                want = 0,
                "correcting motor baud rate register"
            );
            self.with_eeprom_unlocked(id, |c| c.write_baudrate(id, 0))?;
            fixed += 1;
        }
        Ok(fixed)
    }

    /// `phase` bit 2 decides whether a speed count is one step per second or fifty.
    fn read_speed_unit(&mut self) -> Result<()> {
        let id = JOINT_IDS[0];
        let phase = first(self.controller.read_phase(id), "phase", id)?;
        self.speed_steps_per_count = if phase & PHASE_SPEED_UNIT_FINE != 0 {
            1.0
        } else {
            50.0
        };
        tracing::info!(
            phase,
            steps_per_count = self.speed_steps_per_count,
            "servo speed unit"
        );
        Ok(())
    }

    /// Run `write` with the EEPROM lock open, and close it again whatever happened.
    fn with_eeprom_unlocked(
        &mut self,
        id: u8,
        write: impl FnOnce(&mut Sts3215Controller) -> RpResult<()>,
    ) -> Result<()> {
        self.controller
            .write_lock(id, false)
            .map_err(|e| IoError::Bus(format!("unlock eeprom on {id}: {e}")))?;
        let result = write(&mut self.controller);
        std::thread::sleep(EEPROM_SETTLE);
        let relock = self.controller.write_lock(id, true);
        result.map_err(|e| IoError::Bus(format!("eeprom write on {id}: {e}")))?;
        relock.map_err(|e| IoError::Bus(format!("relock eeprom on {id}: {e}")))
    }

    /// The expected servo IDs that do not answer a ping, in [`JOINT_IDS`] order.
    pub fn missing_servos(&mut self) -> Result<Vec<u8>> {
        let mut missing = Vec::new();
        for &id in &JOINT_IDS {
            let answered = self
                .controller
                .ping(id)
                .map_err(|e| IoError::Bus(format!("ping {id}: {e}")))?;
            if !answered {
                missing.push(id);
            }
        }
        Ok(missing)
    }

    /// Flash a factory-fresh servo so it takes the place of the one that is missing.
    ///
    /// A new Feetech servo answers as ID 1 and already at 1 Mbps, so unlike the XL330 path there
    /// is no second baud rate to look at: unlock, write the ID, lock, check, reboot.
    ///
    /// The reboot is kept, for a different reason than on the XL330. There it clears an alert;
    /// here it is the proof that the new ID reached EEPROM — an ID written with the lock still
    /// closed answers until the next power cycle and then is 1 again, and this is the moment to
    /// find that out rather than the next morning.
    pub fn adopt_replacement(&mut self, id: u8) -> Result<bool> {
        let name = JOINT_IDS
            .iter()
            .position(|&j| j == id)
            .map(|i| JOINT_NAMES[i])
            .ok_or_else(|| IoError::Bus(format!("{id} is not a joint id")))?;

        let fresh = self
            .controller
            .ping(FACTORY_ID)
            .map_err(|e| IoError::Bus(format!("ping factory id {FACTORY_ID}: {e}")))?;
        if !fresh {
            return Ok(false);
        }
        tracing::warn!(
            id,
            joint = name,
            "factory-fresh servo on the bus; flashing it as the missing joint"
        );

        // The lock is per servo and the servo changes address half way through: open it under
        // the old ID, close it under the new one.
        self.controller
            .write_lock(FACTORY_ID, false)
            .map_err(|e| IoError::Bus(format!("unlock eeprom on {FACTORY_ID}: {e}")))?;
        self.controller
            .write_id(FACTORY_ID, id)
            .map_err(|e| IoError::Bus(format!("write id {id} on {FACTORY_ID}: {e}")))?;
        std::thread::sleep(EEPROM_SETTLE);
        self.controller
            .write_lock(id, true)
            .map_err(|e| IoError::Bus(format!("relock eeprom on {id}: {e}")))?;

        let fixed = self.check_registers_of(id)?;

        RobotIo::reboot(self, id)?;
        std::thread::sleep(REBOOT_SETTLE);
        let back = self
            .controller
            .ping(id)
            .map_err(|e| IoError::Bus(format!("ping {id} after reboot: {e}")))?;
        if !back {
            return Err(IoError::Bus(format!(
                "servo {id} ({name}) was flashed but did not come back from its reboot — \
                 the id did not reach eeprom, or the firmware has no REBOOT instruction"
            )));
        }
        let status = first(self.controller.read_status(id), "status", id)?;
        if status != 0 {
            tracing::error!(
                id,
                joint = name,
                status,
                "replacement servo still reports a fault after its reboot"
            );
        }
        tracing::warn!(
            id,
            joint = name,
            registers_fixed = fixed,
            "replacement servo adopted"
        );
        Ok(true)
    }

    /// Present positions only, in model radians.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        let blocks = self
            .controller
            .sync_read_raw_data(&JOINT_IDS, READ_ADDR, 2)
            .map_err(|e| IoError::Bus(format!("read present positions: {e}")))?;
        if blocks.len() != NUM_JOINTS {
            return Err(IoError::ShortRead {
                what: "present positions",
                expected: NUM_JOINTS,
                got: blocks.len(),
            });
        }
        let mut out = [0.0; NUM_JOINTS];
        for (joint, block) in blocks.iter().enumerate() {
            if block.len() != 2 {
                return Err(IoError::ShortRead {
                    what: "present position",
                    expected: 2,
                    got: block.len(),
                });
            }
            let raw = u16::from_le_bytes([block[0], block[1]]);
            out[joint] = raw_to_rad(raw, self.config.directions[joint]);
        }
        Ok(out)
    }

    /// Torque on every servo. Every servo is written whatever the others said — see
    /// [`crate::bus::DynamixelIo::set_torque`] for the fall this guards against.
    pub fn set_torque(&mut self, on: bool) -> Result<()> {
        let mut failed = Vec::new();
        for &id in &JOINT_IDS {
            if let Err(e) = self.controller.write_torque_enable(id, on) {
                failed.push(format!("torque {on} on {id}: {e}"));
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            Err(IoError::Bus(failed.join("; ")))
        }
    }

    /// Ramp every joint from where it is now to `target`, linearly. `robotd init` only.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64())
            .ceil()
            .max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let mut next = [0.0; NUM_JOINTS];
            for j in 0..NUM_JOINTS {
                next[j] = start[j] + (target[j] - start[j]) * t;
            }
            self.write(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }

    /// The body IMU's sensor→trunk mount, which is the robot's. Starts the decoder afresh, so
    /// call it right after [`Self::open`], before the first read — the same contract as
    /// [`DynamixelIo::set_imu_mount`], because both feed the same [`SflpDecoder`] and the
    /// `imu_to_dxl` board is the same board either way.
    pub fn set_imu_mount(&mut self, mount: [f64; 4]) {
        self.imu = SflpDecoder::new(mount);
    }

    /// Whether the IMU board is on the bus, asked straight after [`Self::check_registers`].
    ///
    /// Answered with a real combined read rather than a ping: on this bus the board is not a
    /// separate device to interrogate but the last slot of the same `sync_read` the servos
    /// answer, so the honest question is whether that transaction carried its block. It has
    /// already been paid for once by the register check, which cannot have passed with the
    /// board silent — see [`RobotIo::read`], where a short block list is an error.
    ///
    /// Presence, not readiness: [`RobotIo::imu_ready`] counts accepted quaternions and so is
    /// false for the first fraction of a second after the board comes up, which is exactly when
    /// `robotd` asks this. Answering readiness here would make every boot look like an
    /// unplugged board.
    pub fn imu_answers(&mut self) -> Result<bool> {
        self.read().map(|_| true)
    }
}

/// The serial port at `baud`, wrapped in a Feetech-protocol controller.
fn open_controller(port: &str, baud: u32) -> Result<Sts3215Controller> {
    let serial = serialport::new(port, baud)
        .timeout(READ_TIMEOUT)
        .open()
        .map_err(|e| IoError::Port {
            path: port.to_owned(),
            source: std::io::Error::other(e),
        })?;
    Ok(Sts3215Controller::new()
        .with_protocol_v1()
        .with_serial_port(serial))
}

/// rustypot returns a `Vec` even for a single-id read; an empty one means the servo did not
/// answer, which must not be read as "register is fine".
fn first<T: Copy>(read: RpResult<Vec<T>>, name: &'static str, id: u8) -> Result<T> {
    read.map_err(|e| IoError::Bus(format!("read {name} on {id}: {e}")))?
        .first()
        .copied()
        .ok_or(IoError::ShortRead {
            what: "register read",
            expected: 1,
            got: 0,
        })
}

/// Sign-magnitude, bit 15 the sign: how the STS reports position, speed and current.
fn sign_magnitude(raw: u16) -> i32 {
    let magnitude = i32::from(raw & 0x7FFF);
    if raw & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Register count to model radians: 2048 is 0 rad, and `direction` says which way is positive.
///
/// The same mapping as the XL330's — 4096 steps a turn, centre at 2048 — so
/// [`crate::model::DEFAULT_POSITION`] means the same pose on both robots once each joint's
/// zero has been calibrated to the straight-leg pose.
fn raw_to_rad(raw: u16, direction: f64) -> f64 {
    let centred = f64::from(sign_magnitude(raw)) - STEPS_PER_REV / 2.0;
    direction * centred * (2.0 * PI / STEPS_PER_REV)
}

/// Model radians to a register count, clamped to one turn. A target outside it would be
/// read by the servo as a multi-turn move.
fn rad_to_raw(rad: f64, direction: f64) -> u16 {
    let raw = (direction * rad * (STEPS_PER_REV / (2.0 * PI)) + STEPS_PER_REV / 2.0).round();
    raw.clamp(0.0, f64::from(RAW_MAX)) as u16
}

/// XL330-scale gain to the STS `P` byte. Never 0: a zero P is a servo that does not hold, which
/// is not what any caller of `set_gain` means — limp is a low gain, torque off is `set_torque`.
fn scale_gain(kp: u16, p_scale: f64) -> u8 {
    (f64::from(kp) * p_scale).round().clamp(1.0, 254.0) as u8
}

/// One servo's feedback block, decoded.
struct Feedback {
    position: f64,
    velocity: f64,
    current_ma: f64,
    volts: f64,
    temp_c: f64,
}

fn decode_block(block: &[u8], direction: f64, speed_steps_per_count: f64) -> Feedback {
    let position = u16::from_le_bytes([block[0], block[1]]);
    let speed = u16::from_le_bytes([block[2], block[3]]);
    let current = u16::from_le_bytes([block[13], block[14]]);
    Feedback {
        position: raw_to_rad(position, direction),
        velocity: direction
            * f64::from(sign_magnitude(speed))
            * speed_steps_per_count
            * (2.0 * PI / STEPS_PER_REV),
        current_ma: f64::from(sign_magnitude(current)).abs() * MILLIAMPS_PER_COUNT,
        volts: f64::from(block[6]) * VOLTS_PER_COUNT,
        temp_c: f64::from(block[7]),
    }
}

/// The `sync_read` list: the joints in [`JOINT_IDS`] order, then the IMU board **last**.
///
/// The tail position is not a detail. The board answers a fixed delay after the instruction,
/// and that delay is longer than a servo's own — asked first, its reply overlaps the first
/// servo's and both come back corrupt; asked last it answers after every servo has had its
/// turn. Measured on the bench, not theorised: `tools/下游改动.md`, 改动 C.
fn read_ids() -> Vec<u8> {
    let mut ids = Vec::with_capacity(NUM_JOINTS + 1);
    ids.extend_from_slice(&JOINT_IDS);
    ids.push(IMU_DXL_ID);
    ids
}

impl RobotIo for FeetechIo {
    fn read(&mut self) -> Result<Sensors> {
        let blocks = self
            .controller
            .sync_read_raw_data(&self.ids, READ_ADDR, READ_LEN)
            .map_err(|e| IoError::Bus(format!("combined imu+motor sync_read: {e}")))?;
        if blocks.len() != self.ids.len() {
            return Err(IoError::ShortRead {
                what: "sync_read blocks",
                expected: self.ids.len(),
                got: blocks.len(),
            });
        }

        let mut sensors = Sensors::default();

        // The last slot is the IMU board. It answers the servo window and puts its sample in
        // the first IMU_BLOCK_LEN bytes; the three after that are a sample counter, the status
        // bits and a reserved byte, none of which is read yet.
        let imu_block = &blocks[NUM_JOINTS];
        if imu_block.len() != READ_LEN as usize {
            return Err(IoError::ShortRead {
                what: "imu block",
                expected: READ_LEN as usize,
                got: imu_block.len(),
            });
        }
        let mut raw = [0u8; IMU_BLOCK_LEN];
        raw.copy_from_slice(&imu_block[..IMU_BLOCK_LEN]);
        // Say so, or the counters are numbers nobody ever reads — but only once the run is
        // long enough to mean something, and rate-limited past that because a board that has
        // stopped refreshing produces one of these every tick.
        let run = self.stale_imu.observe(&raw);
        if run == STALE_RUN_WARN || (run > STALE_RUN_WARN && run.is_multiple_of(500)) {
            tracing::warn!(
                consecutive = run,
                total = self.stale_imu.stale.total,
                "imu board has returned the same sample {run} reads running — orientation is frozen"
            );
        }
        sensors.imu = self.imu.decode(&raw);

        let mut temps_c = [0.0; NUM_JOINTS];
        let mut volts = Vec::with_capacity(NUM_JOINTS);
        for (joint, block) in blocks[..NUM_JOINTS].iter().enumerate() {
            if block.len() != READ_LEN as usize {
                return Err(IoError::ShortRead {
                    what: "motor block",
                    expected: READ_LEN as usize,
                    got: block.len(),
                });
            }
            let fb = decode_block(
                block,
                self.config.directions[joint],
                self.speed_steps_per_count,
            );
            sensors.positions[joint] = fb.position;
            sensors.velocities[joint] = fb.velocity;
            sensors.currents_ma[joint] = fb.current_ma;
            temps_c[joint] = fb.temp_c;
            // A servo answering 0 V is answering nonsense; it must not drag the mean down as if
            // the pack were half flat.
            if fb.volts > 0.0 {
                volts.push(fb.volts);
            }
        }
        if !volts.is_empty() {
            self.last_slow = Some(SlowSensors {
                volts: volts.iter().sum::<f64>() / volts.len() as f64,
                temps_c,
            });
        }
        Ok(sensors)
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        let data: Vec<Vec<u8>> = targets
            .positions
            .iter()
            .zip(&self.config.directions)
            .map(|(&rad, &dir)| rad_to_raw(rad, dir).to_le_bytes().to_vec())
            .collect();
        self.controller
            .sync_write_raw_data(&JOINT_IDS, GOAL_POSITION_ADDR, &data)
            .map_err(|e| IoError::Bus(format!("sync_write goal positions: {e}")))
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        FeetechIo::set_torque(self, on)
    }

    fn reboot(&mut self, id: u8) -> Result<()> {
        // The protocol document asks for torque off before a REBOOT. Best effort: the servo
        // being rebooted is, by definition, one that may not be answering properly.
        if let Err(e) = self.controller.write_torque_enable(id, false) {
            tracing::warn!(id, error = %e, "could not switch torque off before the reboot");
        }
        // The servo sends no status packet for 0x08; rustypot reports that as `Ok(false)`, so
        // only a failure to send is an error. No waiting here — `Safety::reboot_motors` calls
        // this once per servo from inside the control tick.
        self.controller
            .reboot(id)
            .map(|_| ())
            .map_err(|e| IoError::Bus(format!("reboot {id}: {e}")))
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        // P only, and with the EEPROM lock left closed: the value takes effect and is not
        // persisted, so a reboot or a power cycle restores whatever the servo was tuned to.
        // D and I are deliberately not zeroed as they are on the XL330 — there the factory D
        // is an accident the prototype removed; here it is part of how the vendor tuned a
        // 320:1 gearbox not to ring.
        let p = scale_gain(kp, self.config.p_scale);
        for &id in &JOINT_IDS {
            self.controller
                .write_p_coefficient(id, p)
                .map_err(|e| IoError::Bus(format!("p_coefficient {p} (kp {kp}) on {id}: {e}")))?;
        }
        Ok(())
    }

    /// Supply voltage and case temperatures — from the last tick's block, no transaction.
    ///
    /// Before the first successful read there is nothing cached, so one read is made; that is
    /// the `robotd init` case, which asks before any tick has run.
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        if self.last_slow.is_none() {
            self.read()?;
        }
        self.last_slow.ok_or(IoError::ShortRead {
            what: "input voltage",
            expected: NUM_JOINTS,
            got: 0,
        })
    }

    fn imu_stale(&self) -> ImuStale {
        self.stale_imu.stale
    }

    /// Orientation is a measurement once the board has filled its SFLP window; before that
    /// [`Sensors::imu`] is the identity the decoder starts from, and this is what tells the
    /// rest of the daemon not to treat it as one.
    fn imu_ready(&self) -> bool {
        self.imu.ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2048 is 0 rad and a quarter turn is 1024 counts — the XL330's mapping, which is what
    /// lets `DEFAULT_POSITION` be shared between the two robots.
    #[test]
    fn position_mapping_matches_the_dynamixel_bus() {
        assert_eq!(raw_to_rad(2048, 1.0), 0.0);
        assert!((raw_to_rad(3072, 1.0) - PI / 2.0).abs() < 1e-12);
        assert!((raw_to_rad(1024, 1.0) + PI / 2.0).abs() < 1e-12);
        for raw in [0u16, 1, 1024, 2047, 2048, 2049, 3072, 4095] {
            assert_eq!(rad_to_raw(raw_to_rad(raw, 1.0), 1.0), raw);
            assert_eq!(rad_to_raw(raw_to_rad(raw, -1.0), -1.0), raw);
        }
    }

    /// A reversed joint mirrors about 2048 — it must not mirror about 0, which would send a
    /// knee asked for +10° to the far end of its travel.
    #[test]
    fn a_reversed_joint_mirrors_about_the_centre() {
        let ten_degrees = 10.0_f64.to_radians();
        let forward = rad_to_raw(ten_degrees, 1.0);
        let reversed = rad_to_raw(ten_degrees, -1.0);
        assert_eq!(i32::from(forward) - 2048, 2048 - i32::from(reversed));
        assert!((raw_to_rad(reversed, -1.0) - ten_degrees).abs() < 2.0 * PI / STEPS_PER_REV);
    }

    /// Targets past one turn are clamped rather than wrapped: the servo would take a wrapped
    /// value as a request to go the long way round.
    #[test]
    fn targets_outside_one_turn_are_clamped() {
        assert_eq!(rad_to_raw(10.0, 1.0), 4095);
        assert_eq!(rad_to_raw(-10.0, 1.0), 0);
    }

    /// Bit 15 is a sign, not a value bit. Read as plain `u16`, a slow reverse speed of 5
    /// counts would arrive as 32773 — a joint doing thirty radians a second.
    #[test]
    fn bit_fifteen_is_the_sign() {
        assert_eq!(sign_magnitude(5), 5);
        assert_eq!(sign_magnitude(0x8005), -5);
        assert_eq!(sign_magnitude(0x8000), 0);
    }

    fn block(position: u16, speed: u16, volts: u8, temp: u8, current: u16) -> Vec<u8> {
        let mut b = vec![0u8; READ_LEN as usize];
        b[0..2].copy_from_slice(&position.to_le_bytes());
        b[2..4].copy_from_slice(&speed.to_le_bytes());
        b[6] = volts;
        b[7] = temp;
        b[13..15].copy_from_slice(&current.to_le_bytes());
        b
    }

    /// Every field of the block lands where the register table says, in SI units.
    #[test]
    fn a_feedback_block_decodes_into_si_units() {
        let fb = decode_block(&block(3072, 0x8000 | 100, 74, 36, 20), 1.0, 1.0);
        assert!((fb.position - PI / 2.0).abs() < 1e-12);
        assert!((fb.velocity + 100.0 * 2.0 * PI / 4096.0).abs() < 1e-12);
        assert!((fb.current_ma - 130.0).abs() < 1e-9);
        assert!((fb.volts - 7.4).abs() < 1e-9);
        assert_eq!(fb.temp_c, 36.0);
    }

    /// Direction flips velocity along with position; one without the other is a joint whose
    /// observed velocity opposes its observed motion, which a policy reads as a disturbance.
    #[test]
    fn direction_flips_velocity_with_position() {
        let forward = decode_block(&block(2148, 100, 74, 30, 0), 1.0, 1.0);
        let reversed = decode_block(&block(2148, 100, 74, 30, 0), -1.0, 1.0);
        assert_eq!(forward.position, -reversed.position);
        assert_eq!(forward.velocity, -reversed.velocity);
    }

    /// With `phase` bit 2 clear a speed count is fifty steps a second, not one.
    #[test]
    fn the_coarse_speed_unit_is_fifty_times_the_fine_one() {
        let fine = decode_block(&block(2048, 3, 74, 30, 0), 1.0, 1.0);
        let coarse = decode_block(&block(2048, 3, 74, 30, 0), 1.0, 50.0);
        assert!((coarse.velocity - 50.0 * fine.velocity).abs() < 1e-12);
    }

    /// The daemon's three gains at the default scale: running 200 is the factory 32, and the
    /// limp gain is soft but not zero.
    #[test]
    fn gains_scale_onto_one_byte_and_never_reach_zero() {
        assert_eq!(scale_gain(200, 0.16), 32);
        assert_eq!(scale_gain(160, 0.16), 26);
        assert_eq!(scale_gain(50, 0.16), 8);
        assert_eq!(scale_gain(0, 0.16), 1);
        assert_eq!(scale_gain(u16::MAX, 0.16), 254);
    }

    /// The board must be asked **last**. Asked first it collides with the first servo's own
    /// reply and corrupts both — measured, not theorised (`tools/下游改动.md`, 改动 C).
    #[test]
    fn the_imu_board_is_asked_last() {
        let ids = read_ids();
        assert_eq!(ids.len(), NUM_JOINTS + 1);
        assert_eq!(&ids[..NUM_JOINTS], &JOINT_IDS);
        assert_eq!(ids[NUM_JOINTS], IMU_DXL_ID);
        assert!(!ids[..NUM_JOINTS].contains(&IMU_DXL_ID));
    }

    #[test]
    fn config_refuses_directions_that_are_not_plus_or_minus_one() {
        let mut d = [1i8; NUM_JOINTS];
        assert!(FeetechConfig::new(0.16, &d).is_ok());
        d[3] = -1;
        assert_eq!(FeetechConfig::new(0.16, &d).unwrap().directions[3], -1.0);
        d[3] = 0;
        assert!(FeetechConfig::new(0.16, &d).is_err());
        assert!(FeetechConfig::new(0.16, &d[..14]).is_err());
        assert!(FeetechConfig::new(0.0, &[1i8; NUM_JOINTS]).is_err());
    }
}
