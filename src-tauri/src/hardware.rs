//! src-tauri/src/hardware.rs
//! Lightweight Hardware Telemetry Engine in Rust.
//! Reads CPU, RAM, and NVIDIA GPU statistics without importing PyTorch or Python.

use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTelemetry {
    pub cuda_available: bool,
    pub has_gpu: bool,
    pub gpu_name: String,
    pub device_name: String,
    pub gpu_count: usize,
    pub memory_used_mb: f64,
    pub used_vram_mb: f64,
    pub memory_total_mb: f64,
    pub total_vram_mb: f64,
    pub memory_free_mb: f64,
    pub free_vram_mb: f64,
    pub memory_utilization_pct: f32,
    pub vram_percent: f32,
    pub gpu_utilization_pct: f32,
    pub gpu_util_percent: f32,
    pub gpu_utilization: f32,
    pub temperature_c: f32,
    pub gpu_temperature_c: f32,
    pub power_draw_w: f32,
    pub power_limit_w: f32,
    pub core_clock_mhz: f32,
    pub sm_clock_mhz: f32,
    pub memory_clock_mhz: f32,
    pub mem_clock_mhz: f32,
    pub cpu_percent: f32,
    pub ram_used_mb: f64,
    pub ram_total_mb: f64,
    pub ram_percent: f32,
    pub system_ram_used_mb: f64,
    pub system_ram_total_mb: f64,
    pub system_ram_free_mb: f64,
    pub system_ram_used_gb: f64,
    pub system_ram_total_gb: f64,
    pub timestamp: f64,
}

pub struct HardwareMonitor {
    sys: System,
}

impl HardwareMonitor {
    pub fn new() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_all();
        Self { sys }
    }

    pub fn read_telemetry(&mut self) -> HardwareTelemetry {
        self.sys.refresh_cpu();
        self.sys.refresh_memory();

        let cpus = self.sys.cpus();
        let cpu_percent = if !cpus.is_empty() {
            cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
        } else {
            0.0
        };

        let ram_used_mb = self.sys.used_memory() as f64 / (1024.0 * 1024.0);
        let ram_total_mb = self.sys.total_memory() as f64 / (1024.0 * 1024.0);
        let ram_free_mb = (ram_total_mb - ram_used_mb).max(0.0);
        let ram_percent = if ram_total_mb > 0.0 {
            ((ram_used_mb / ram_total_mb) * 100.0) as f32
        } else {
            0.0
        };

        let system_ram_used_gb = ram_used_mb / 1024.0;
        let system_ram_total_gb = ram_total_mb / 1024.0;

        let (has_gpu, gpu_name, vram_used, vram_total, gpu_util, gpu_temp, pwr_draw, pwr_limit, sm_clk, mem_clk) =
            Self::query_nvidia_smi();

        let vram_free = (vram_total - vram_used).max(0.0);
        let vram_pct = if vram_total > 0.0 {
            ((vram_used / vram_total) * 100.0) as f32
        } else {
            0.0
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        HardwareTelemetry {
            cuda_available: has_gpu,
            has_gpu,
            gpu_name: gpu_name.clone(),
            device_name: gpu_name,
            gpu_count: if has_gpu { 1 } else { 0 },
            memory_used_mb: vram_used,
            used_vram_mb: vram_used,
            memory_total_mb: vram_total,
            total_vram_mb: vram_total,
            memory_free_mb: vram_free,
            free_vram_mb: vram_free,
            memory_utilization_pct: vram_pct,
            vram_percent: vram_pct,
            gpu_utilization_pct: gpu_util,
            gpu_util_percent: gpu_util,
            gpu_utilization: gpu_util,
            temperature_c: gpu_temp,
            gpu_temperature_c: gpu_temp,
            power_draw_w: pwr_draw,
            power_limit_w: pwr_limit,
            core_clock_mhz: sm_clk,
            sm_clock_mhz: sm_clk,
            memory_clock_mhz: mem_clk,
            mem_clock_mhz: mem_clk,
            cpu_percent,
            ram_used_mb,
            ram_total_mb,
            ram_percent,
            system_ram_used_mb: ram_used_mb,
            system_ram_total_mb: ram_total_mb,
            system_ram_free_mb: ram_free_mb,
            system_ram_used_gb,
            system_ram_total_gb,
            timestamp: now,
        }
    }

    fn query_nvidia_smi() -> (bool, String, f64, f64, f32, f32, f32, f32, f32, f32) {
        #[cfg(target_os = "windows")]
        let mut cmd = Command::new("nvidia-smi");
        #[cfg(target_os = "windows")]
        cmd.creation_flags(CREATE_NO_WINDOW);

        #[cfg(not(target_os = "windows"))]
        let mut cmd = Command::new("nvidia-smi");

        cmd.args(&[
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ]);

        if let Ok(output) = cmd.output() {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = text.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 9 {
                        let name = parts[0].to_string();
                        let used_mb = parts[1].parse::<f64>().unwrap_or(0.0);
                        let total_mb = parts[2].parse::<f64>().unwrap_or(6144.0);
                        let util = parts[3].parse::<f32>().unwrap_or(0.0);
                        let temp = parts[4].parse::<f32>().unwrap_or(0.0);
                        let pwr_draw = parts[5].parse::<f32>().unwrap_or(0.0);
                        let pwr_limit = parts[6].parse::<f32>().unwrap_or(80.0);
                        let sm_clk = parts[7].parse::<f32>().unwrap_or(0.0);
                        let mem_clk = parts[8].parse::<f32>().unwrap_or(0.0);

                        return (
                            true,
                            name,
                            used_mb,
                            total_mb,
                            util,
                            temp,
                            pwr_draw,
                            pwr_limit,
                            sm_clk,
                            mem_clk,
                        );
                    }
                }
            }
        }

        (false, "No GPU Detected".to_string(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}
