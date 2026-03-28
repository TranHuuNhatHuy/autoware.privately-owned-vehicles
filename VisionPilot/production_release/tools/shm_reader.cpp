/**
 * @file shm_reader.cpp
 * @brief Quick smoke-test: attach to /visionpilot_state and print every update.
 *
 * Build:
 *   g++ -std=c++17 -O2 -o shm_reader tools/shm_reader.cpp -lrt
 *
 * Run (while visionpilot is running):
 *   ./shm_reader            # prints every new frame
 *   ./shm_reader --once     # prints one snapshot and exits
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <csignal>

// ---- mirror of VisionPilotState (must stay in sync with the header) -------
struct VisionPilotState {
    volatile uint64_t seq;
    uint64_t frame_number;

    // Lateral
    double steering_pid_deg;
    double steering_pid_raw_deg;
    double steering_autosteer_deg;
    bool   autosteer_valid;
    double cte_m;
    double yaw_error_rad;
    double curvature_inv_m;
    bool   path_valid;
    bool   lane_departure_warning;

    // Longitudinal
    bool   cipo_exists;
    int    cipo_track_id;
    int    cipo_class_id;
    double cipo_distance_m;
    double cipo_velocity_ms;
    bool   cut_in_detected;
    bool   kalman_reset;
    double ideal_speed_ms;
    double safe_distance_m;
    bool   fcw_active;
    bool   aeb_active;
    double control_effort_ms2;

    // CAN / ego
    double ego_speed_ms;
    double ego_steering_angle_deg;
    bool   can_valid;
};

// ---- seqlock read (same logic as VisionPilotSharedState::read) ------------
static inline uint64_t seq_load(const volatile uint64_t* p) {
    uint64_t v;
    __atomic_load(p, &v, __ATOMIC_ACQUIRE);
    return v;
}

static void seqlock_read(const VisionPilotState* src, VisionPilotState& out) {
    uint64_t s1, s2;
    do {
        s1 = seq_load(&src->seq);
        if (s1 & 1u) continue;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        std::memcpy(&out, src, sizeof(VisionPilotState));
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        s2 = seq_load(&src->seq);
    } while (s1 != s2);
}

static volatile bool g_running = true;
void on_sigint(int) { g_running = false; }

// ---------------------------------------------------------------------------
static void print_state(const VisionPilotState& s) {
    std::cout << "\033[2J\033[H";  // clear terminal
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "========== VisionPilot Shared State  [frame " << s.frame_number << "] ==========\n";

    std::cout << "\n--- LATERAL ---\n";
    std::cout << "  Steering PID (filtered):  " << std::setw(8) << s.steering_pid_deg       << " deg\n";
    std::cout << "  Steering PID (raw):       " << std::setw(8) << s.steering_pid_raw_deg   << " deg\n";
    std::cout << "  AutoSteer model:          " << std::setw(8) << s.steering_autosteer_deg << " deg"
              << (s.autosteer_valid ? "" : "  [not ready]") << "\n";
    std::cout << "  PathFinder valid:         " << (s.path_valid ? "YES" : "NO") << "\n";
    if (s.path_valid) {
        std::cout << "    CTE:                    " << std::setw(8) << s.cte_m           << " m\n";
        std::cout << "    Yaw error:              " << std::setw(8) << s.yaw_error_rad   << " rad\n";
        std::cout << "    Curvature:              " << std::setw(8) << s.curvature_inv_m << " 1/m\n";
    }
    std::cout << "  Lane departure warning:   " << (s.lane_departure_warning ? "\033[1;33mWARNING\033[0m" : "ok") << "\n";

    std::cout << "\n--- LONGITUDINAL ---\n";
    if (s.cipo_exists) {
        std::cout << "  CIPO:  Track " << s.cipo_track_id
                  << "  Class " << s.cipo_class_id
                  << "  dist=" << std::setw(7) << s.cipo_distance_m  << " m"
                  << "  vel="  << std::setw(7) << s.cipo_velocity_ms << " m/s"
                  << (s.cut_in_detected ? "  \033[1;35m[CUT-IN]\033[0m" : "")
                  << (s.kalman_reset    ? "  [KALMAN RESET]" : "")
                  << "\n";
        std::cout << "  RSS d_safe:               " << std::setw(8) << s.safe_distance_m << " m\n";
        std::cout << "  Ideal speed:              " << std::setw(8) << s.ideal_speed_ms  << " m/s\n";

        std::string fcw_str = s.aeb_active ? "\033[1;31m!!! AEB !!!\033[0m"
                            : s.fcw_active ? "\033[1;33m! FCW !\033[0m"
                            :                "\033[0;32mOK\033[0m";
        std::cout << "  Safety:                   " << fcw_str << "\n";
        std::cout << "  Control effort:           " << std::setw(8) << s.control_effort_ms2 << " m/s²"
                  << (s.control_effort_ms2 >= 0 ? "  [ACCEL]" : "  [BRAKE]") << "\n";
    } else {
        std::cout << "  CIPO:  none\n";
        std::cout << "  Control effort:           " << std::setw(8) << s.control_effort_ms2 << " m/s²\n";
    }

    std::cout << "\n--- CAN / EGO ---\n";
    if (s.can_valid) {
        std::cout << "  Ego speed:                " << std::setw(8) << s.ego_speed_ms           << " m/s\n";
        std::cout << "  Ego steering angle:       " << std::setw(8) << s.ego_steering_angle_deg << " deg\n";
    } else {
        std::cout << "  CAN data:  not available\n";
    }

    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool once = (argc > 1 && std::string(argv[1]) == "--once");

    signal(SIGINT, on_sigint);

    int fd = shm_open("/visionpilot_state", O_RDONLY, 0);
    if (fd < 0) {
        perror("shm_open");
        std::cerr << "Is VisionPilot running?\n";
        return 1;
    }

    void* ptr = mmap(nullptr, sizeof(VisionPilotState), PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    const auto* shm = reinterpret_cast<const VisionPilotState*>(ptr);
    std::cout << "[shm_reader] Attached to /visionpilot_state (" << sizeof(VisionPilotState) << " bytes)\n";

    uint64_t last_frame = UINT64_MAX;

    while (g_running) {
        VisionPilotState snap{};
        seqlock_read(shm, snap);

        if (snap.frame_number != last_frame) {
            last_frame = snap.frame_number;
            print_state(snap);
            if (once) break;
        }

        usleep(50000);  // poll at ~20 Hz (visionpilot runs at 10 Hz, no need to spin)
    }

    munmap(ptr, sizeof(VisionPilotState));
    close(fd);
    std::cout << "\n[shm_reader] detached.\n";
    return 0;
}
