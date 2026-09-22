# GeoSR-Varanasi-SIH
# GeoSR — Live Demo (SIH 26142, Team Varanasi)

Full-stack demo matching the tools named in your pitch deck:

| Slide tool | Used as |
|---|---|
| PyTorch, EDSR | `backend/model.py` — EDSR-lite residual SR network |
| OpenCV/PIL, NumPy | image I/O + preprocessing in `backend/utils.py` |
| Flask | `backend/app.py` — `/api/infer` inference endpoint |
| Docker | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| React.js | `frontend/` dashboard (upload → compare → metrics) |
| Mapbox GL | swapped for **Leaflet** in the demo (identical UX, zero API key needed — see note in `MapView.jsx` for swapping back to Mapbox) |
| Scikit-image (PSNR/SSIM) | `backend/utils.py: compute_metrics()` |

## Why the model is real, not a mockup

`GeoSR` (in `model.py`) is a genuine trainable residual super-resolution
network (head conv → 6 residual blocks → pixel-shuffle upsampler), the
same architecture family as EDSR, just scaled down so it trains on a
handful of images on a laptop in minutes instead of needing a GPU
cluster and the full Sentinel-2 archive. It is not bicubic interpolation
dressed up — you can see it out-perform the bicubic baseline on PSNR/SSIM
in the dashboard once trained.

## 1. Add your images and train

```bash
cd backend
pip install -r requirements.txt

# Drop your uploaded Sentinel-2 / satellite tiles here (jpg/png/tif):
mkdir -p data/hr_images
cp /path/to/your/satellite_tiles/*.* data/hr_images/

python train.py --data_dir ./data/hr_images --epochs 30 --scale 4
# -> saves backend/checkpoints/geosr.pt
```

Training synthesizes LR/HR pairs by downsampling *your own* images by
the scale factor — the standard recipe used by SRCNN/EDSR/ESRGAN papers
(swap in real paired 10m/<4m tiles for production). More images and more
epochs = sharper results; 5-10 images and 30-50 epochs is enough for a
convincing live demo.

## 2. Run the backend

```bash
cd backend
python app.py
# Flask API on http://localhost:5000  (GET /api/health to check it loaded your checkpoint)
```

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard on http://localhost:5173
```

## 4. Or run both with Docker

```bash
docker compose up --build
```

## Live demo script (for judges)

1. Open the dashboard, point at the **Area of Interest** map (Varanasi AOI).
2. Drop a held-out medium-res tile in the left panel + (optionally) its
   true high-res counterpart as the reference.
3. Click **Run Super-Resolution** — Flask calls the trained PyTorch model,
   returns the reconstructed tile + a bicubic baseline for comparison.
4. Drag the before/after slider to show buildings/roads/field edges
   sharpening up. Toggle **Bicubic vs GeoSR** to show the model is doing
   more than simple upsampling.
5. Point at the PSNR/SSIM cards — GeoSR should beat the bicubic baseline
   on both, which is your quantitative "validation-driven" claim from
   the deck.

## Notes / honest limitations to mention if asked

- Training used your uploaded sample tiles as both source and synthesized
  ground truth (self-supervised downsample/upsample recipe) — not a
  paired real 10m→<4m Sentinel-2 dataset, which SIH judges will
  understand isn't publicly available at hackathon scale.
- For production you'd swap in ESA's Sentinel-2 + a real high-res
  reference source (see slide 6 links: ESAOpenSR/SEN2SR, LDSR-S2) and
  train on a GPU for far longer.
