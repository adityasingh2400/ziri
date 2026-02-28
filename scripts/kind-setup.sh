#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="ziri-local"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Creating kind cluster: $CLUSTER_NAME"
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "    Cluster already exists, skipping creation."
else
    kind create cluster --name "$CLUSTER_NAME" --config - <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 30080
        protocol: TCP
EOF
fi

echo "==> Building Docker images"
docker build -t ziri-api:latest   -f "$PROJECT_DIR/docker/api.Dockerfile"    "$PROJECT_DIR"
docker build -t ziri-worker:latest -f "$PROJECT_DIR/docker/worker.Dockerfile" "$PROJECT_DIR"

echo "==> Loading images into kind cluster"
kind load docker-image ziri-api:latest    --name "$CLUSTER_NAME"
kind load docker-image ziri-worker:latest --name "$CLUSTER_NAME"

echo "==> Applying Kubernetes manifests"
kubectl apply -k "$PROJECT_DIR/k8s/"

echo "==> Waiting for pods to be ready"
kubectl -n ziri rollout status statefulset/postgres --timeout=120s || true
kubectl -n ziri rollout status deployment/worker    --timeout=120s || true
kubectl -n ziri rollout status deployment/api       --timeout=120s || true

echo ""
echo "==> Cluster is ready!"
echo "    API:  http://localhost:30080"
echo "    Pods: kubectl -n ziri get pods"
