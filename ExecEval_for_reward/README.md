
---

## 🐳 Docker Usage

You can run the code execution environment using Docker.

### ✅ Step 1: Docker Compose

Build and start the service:

```bash
docker compose up --build
```

---

### ✅ Step 2: Docker Run 

Run the container directly:

```bash
docker run --privileged -it -p 5628:5000 -e NUM_WORKERS=67 exec-eval:1.0
```

Or with a container name:

```bash
docker run --privileged -it \
  -p 7789:5000 \
  -e NUM_WORKERS=67 \
  --name exec-eval-container1 \
  exec-eval:1.0
```
