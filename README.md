# Object Detection Microservice

A microservice architecture with a UI backend and AI backend. The UI backend accepts image uploads from users, forwards them to the AI backend which uses YOLOv3 for object detection, and returns results in structured JSON format with bounding boxes.

## Install Docker

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y docker.io docker-compose
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker $USER  # Log out and back in after this
```

**macOS/Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## Quick Start

```bash
docker-compose up --build -d
```
Open

[http://localhost:3000](http://localhost:3000)
## Services
```

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Web UI |
| UI Backend | 5000 | REST API |
| AI Backend | 5001 | YOLOv3 inference |

## Usage

1. Open http://localhost:3000
2. Upload an image (drag & drop or click)
3. Click "Detect Objects"
4. View results with bounding boxes
```
## Commands

```
docker-compose logs -f          # View logs
docker-compose down             # Stop
docker-compose down --rmi all --volumes && docker-compose up --build -d  # Clean restart
```

## Troubleshooting

**Port in use:**
```bash
for port in 3000 5000 5001; do sudo lsof -ti:$port | xargs -r kill -9; done
```

## Solution

### Approach

Built a microservice architecture with three components:
1. **Frontend** (React) - User interface for image upload
2. **UI Backend** (Flask) - REST API handling requests
3. **AI Backend** (Flask + OpenCV) - YOLOv3 object detection

### Steps

1. Set up Docker containers for each service
2. Implemented image upload endpoint in UI backend
3. Integrated YOLOv3 model in AI backend for object detection
4. Connected services via Docker network
5. Returned detection results as JSON with bounding box coordinates
6. Saved output images with drawn bounding boxes

### Output Format

Detection results saved in `outputs/` folder:
- `*_detected.jpg` - Images with bounding boxes
- `*_detection.json` - JSON with detection data (class, confidence, coordinates)

### References

- [YOLOv3](https://pjreddie.com/darknet/yolo/)
- [OpenCV DNN module](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
- [Flask documentation](https://flask.palletsprojects.com/)
- [Docker Compose](https://docs.docker.com/compose/)
