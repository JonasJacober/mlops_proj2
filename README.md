# Project 2 – Containerization (Starter)

This repo demonstrates a reproducible ML training workflow in **Docker** and **GitHub Codespaces**.

---


### Local Docker build & run

```bash
# Build the Docker image
docker build -t ml-trainer .

# Run a training job (override defaults from Dockerfile CMD)
docker run --rm -it -v "$(pwd)":/workspace ml-trainer \
  --checkpoint_dir models --lr 1e-3 --epochs 5
```

### View TensorBoard
After training completes:
```bash
python -m tensorboard.main --logdir runs --port 6006
```
Then open http://localhost:6006

---

## Run in GitHub Codespaces

This repo includes a `.devcontainer/devcontainer.json` that builds your Codespace **from your Dockerfile**.

1. Push this repo to GitHub  
2. Go to **Code → Create codespace on main**  
3. Once built, run inside the Codespace terminal:

   ```bash
   python main.py --checkpoint_dir models --lr 1e-3 --epochs 5
   ```

4. To view TensorBoard:
   - Forward port **6006** in the “Ports” panel  
   - Click the forwarded URL to open TensorBoard in your browser

---

## Customize

- Replace `main.py` with your actual ML training script  
- Update `requirements.txt` with your real dependencies  
- If you need Docker CLI inside the codespace, use a `docker-in-docker` feature variant instead of the current devcontainer setup  

---

## Reproducibility

- Deterministic seeds via `--seed` argument  
- Containerized dependencies

---

## License
MIT
