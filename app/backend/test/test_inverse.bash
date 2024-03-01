echo "Test inverse project time series data route"
test_route=$(curl -s -X POST "http://localhost:8000/inverse_project_activations/" -H "Content-Type: application/json" -d '{"data": [[9.50953395952177, 2.037727718926309]]}')
if [ "$(echo "$test_route" | wc -l)" -eq 0 ]; then
    echo "Route not working"
else
    echo "${test_route:0:100}"
fi

test_route=$(curl -s -X POST "http://localhost:8000/inverse_project_activations/" -H "Content-Type: application/json" -d '{"data": [[11.081429411400828, 3.4932264402453943]]}')
if [ "$(echo "$test_route" | wc -l)" -eq 0 ]; then
    echo "Route not working"
else
    echo "${test_route:0:100}"
fi
