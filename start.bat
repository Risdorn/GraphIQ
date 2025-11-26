echo "Starting backend application..."
start cmd /k "cd backend && python main.py"

echo "Starting frontend application..."
start cmd /k "cd frontend && pnpm dev"

echo "All services started successfully!"