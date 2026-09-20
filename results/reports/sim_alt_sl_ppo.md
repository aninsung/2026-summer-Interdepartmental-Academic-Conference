# Alternating SL <-> PPO simulation

Pipeline unchanged. Synthetic data. Rounds=3.

Each round: SL Fix -> PPO on Fix inits -> success bank -> SL retrain

| Size | Round | Fix DSC | Hybrid DSC | dHybrid vs Rough | bank |
|---|---:|---:|---:|---:|---:|
| small | 1 | 0.5092 | 0.8226 | +0.3842 | 24 |
| small | 2 | 0.8399 | 0.8357 | +0.3974 | 37 |
| small | 3 | 0.8520 | 0.8555 | +0.4172 | 51 |
| small | summary | 0.5092->0.8520 | 0.8226->0.8555 | hybrid_delta=+0.0329 | sl=2s ppo=44s |
| medium | 1 | 0.8315 | 0.9267 | +0.0952 | 24 |
| medium | 2 | 0.9375 | 0.9370 | +0.1055 | 29 |
| medium | 3 | 0.9351 | 0.9341 | +0.1026 | 29 |
| medium | summary | 0.8315->0.9351 | 0.9267->0.9341 | hybrid_delta=+0.0074 | sl=2s ppo=86s |
| large | 1 | 0.9328 | 0.9601 | +0.0282 | 24 |
| large | 2 | 0.9428 | 0.9640 | +0.0322 | 48 |
| large | 3 | 0.9643 | 0.9643 | +0.0325 | 48 |
| large | summary | 0.9328->0.9643 | 0.9601->0.9643 | hybrid_delta=+0.0042 | sl=2s ppo=88s |

Total: 256.6s

Note: relative round trends only.
