from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

print("1. Testing GET /api/health...")
res = client.get("/api/health")
assert res.status_code == 200, res.text
print("   -> OK:", res.json())

print("2. Testing GET /api/meta...")
res = client.get("/api/meta")
assert res.status_code == 200, res.text
meta = res.json()
print(f"   -> OK: {len(meta['products'])} products, aspect taxonomy: {meta['aspect_taxonomy']}")

print("3. Testing GET /api/presets...")
res = client.get("/api/presets")
assert res.status_code == 200, res.text
presets = res.json()
print(f"   -> OK: {len(presets)} scenario presets")

print("4. Testing GET /api/dashboard-data?dataset=kaggle...")
res = client.get("/api/dashboard-data?dataset=kaggle")
assert res.status_code == 200, res.text
dash = res.json()
print("   -> OK: KPIs:", dash["kpis"])

print("5. Testing POST /api/analyze (5-star with 2 Goods, 1 Bad)...")
res = client.post("/api/analyze", json={
    "text": "The camera photo quality is stunning and the screen display is gorgeous, though battery drains a bit fast.",
    "actual_rating": 5.0,
    "prior_rating_mean": 4.1
})
assert res.status_code == 200, res.text
an = res.json()
print(f"   -> OK: Predicted: {an['predicted_stars']} stars | Actual: {an['actual_stars']} stars | Goods: {an['good_aspects']} | Bads: {an['bad_aspects']}")
print(f"   -> Diagnosis: {an['diagnosis']['title']}")

print("6. Testing GET / (React SPA delivery)...")
res = client.get("/")
assert res.status_code == 200, res.text
assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()
print("   -> OK: React index.html served correctly")

print("\nALL API ENDPOINTS PASSED VERIFICATION!")
