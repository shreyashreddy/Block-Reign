# Block-Reign — Player vs AI Grid Battle 🕹️🤖

## Overview
**Block-Reign** is a lightweight Player vs AI grid battle game where the AI learns from every match you play. The project includes a simple Q-learning AI (used by the game server) and a more advanced DQN trainer for experiments.

---

## 🔑 Key Features
- 🎮 **10×10 grid combat** with movement and shooting
- 🤖 **Self-learning AI** that adapts from actual matches you play
- 💾 **Persistent training** (saved automatically under `training/models`)
- 🧠 **Advanced DQN trainer** available in `ai_trainer.py` for experimentation
- 🛠️ **Easy reset** — delete `training/` to get a fresh (untrained) AI

---

## Current AI Progress ✅
- **Win rate:** 59%
- **Latest match log:** 🏁 Match 78 complete - Winner: ai
- **Recent learning result:** ✅ AI learned from match. Wins: 46, Losses: 36

> These numbers reflect the Q-learning AI's current record (persisted in `training/models/simple_ai.pkl`).

---

## Quick Start — Play Locally (Windows / macOS / Linux)
1. Install Python 3.8+ and pip.
2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the game server:

```bash
python3 game_server.py
```

5. Open your browser at: http://localhost:5000

- **Controls:** Arrow Keys = Move, Space = Shoot

---

## How Learning Works 💡
- The running server uses a simple Q-learning AI (`SimpleAI` in `game_server.py`) that stores a Q-table and tracks wins/losses.
- After each completed match, the client POSTs match data to `/learn_from_match` and the AI updates its policy on-disk.
- The project includes `ai_trainer.py` (a DQN-based trainer) for more advanced experiments and batch training.

---

## Resetting to an Untrained AI 🔄
If you want to start over and play against a brand-new AI:

- Delete the entire `training/` folder (destructive):

```bash
rm -rf training/
```

- Or remove only the saved model:

```bash
rm training/models/simple_ai.pkl
```

After removal, start the server and play matches — the AI will begin learning from scratch as you play.

> ⚠️ Deleting `training/` will remove all saved models, replay buffers and saved checkpoints. Make backups if needed.

---

## Development & Troubleshooting 🔧
- Logs and training data are saved under `training/models`.
- To inspect AI stats during runtime, look for prints from `SimpleAI` at server startup and printed match summaries.
- If models fail to load, ensure `training/models` exists and is writable by your user.

---

## Contributing
Contributions welcome! Fork, make changes, and open a PR. If you're adding features, please include tests and update this README.

---

## License
This project is licensed under the project LICENSE in the repository.

---

Enjoy playing — and see how quickly the AI learns from your style! 🎯
