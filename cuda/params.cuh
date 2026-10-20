// Model constants. Values and names mirror crates/flock-ref/src/params.rs,
// which copies murmur-ref/src/params.rs from murmuration-gpu (MIT).
// docs/SPEC.md lists both sides; keep them identical.
#pragma once

constexpr float DT = 0.005f;
constexpr float DU = 0.05f;
constexpr float V0 = 10.0f;
constexpr float MASS = 0.08f;
constexpr float CL_CD = 3.3f;
constexpr float G = 9.81f;
constexpr float L0 = MASS * G;
constexpr float D0 = L0 / CL_CD;
constexpr float TAU = 1.0f;
constexpr float TAU_ROLL = 0.15f;
constexpr float MAX_ROLL_RATE = 6.0f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float MAX_BANK = 80.0f * PI_F / 180.0f;
constexpr float V_MIN = 0.5f * V0;
constexpr float V_MAX = 2.0f * V0;
constexpr float N_C = 6.5f;
constexpr float R_MAX = 100.0f;
constexpr float RADIUS_LERP = 0.1f;
constexpr float R_H = 0.2f;
constexpr float SIGMA = 1.37f;
constexpr float W_SEP = 1.0f;
constexpr float W_ALIGN = 0.5f;
constexpr float W_COH = 0.65f;
constexpr float BLIND_COS = -0.70710678118654752440f;
constexpr float W_ROOST_V = 0.2f;
constexpr float H_ROOST_V = 40.0f;
constexpr float Z0 = 100.0f;
constexpr float GRID_CELL = 4.0f;
