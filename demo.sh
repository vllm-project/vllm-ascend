if pytest --collect-only -m "multi" | grep "collected 0 items"; then
	echo "No matching tests found. Skipping..."
	exit 0
fi
echo "test"
