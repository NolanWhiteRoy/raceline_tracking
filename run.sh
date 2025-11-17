#!/bin/bash
# SE380 Raceline Tracking Controller - Run Script

echo "================================"
echo "SE380 Raceline Tracking"
echo "================================"

# Check if Python dependencies are installed
echo "Checking dependencies..."
python3 -c "import numpy; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install --user -r requirements.txt
fi

# Parse command line arguments
if [ "$1" == "montreal" ]; then
    echo "Running Montreal track..."
    python3 main.py ./racetracks/Montreal.csv ./racetracks/Montreal_raceline.csv
elif [ "$1" == "ims" ]; then
    echo "Running IMS track..."
    python3 main.py ./racetracks/IMS.csv ./racetracks/IMS_raceline.csv
elif [ "$1" == "test" ]; then
    echo "Running test suite..."
    python3 test_runner.py
elif [ "$1" == "tune" ]; then
    if [ "$2" == "montreal" ] || [ "$2" == "ims" ]; then
        echo "Tuning controller for $2..."
        python3 test_runner.py tune $2
    else
        echo "Usage: ./run.sh tune [montreal|ims]"
    fi
else
    echo "Usage: ./run.sh [montreal|ims|test|tune]"
    echo ""
    echo "Options:"
    echo "  montreal - Run simulation on Montreal track"
    echo "  ims      - Run simulation on IMS track"
    echo "  test     - Run automated tests on all tracks"
    echo "  tune     - Tune controller parameters"
    echo ""
    echo "Default: Running Montreal track..."
    python3 main.py ./racetracks/Montreal.csv ./racetracks/Montreal_raceline.csv
fi
