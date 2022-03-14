echo "Since the two training processes take too long a time, directly download the trained paramaters for future use"
echo "Please ensure you have downloaded the trained parameters from Google drive"

echo "Unzip all needed data files for processing"
unzip data.zip
rm data.zip

unzip processed.zip
rm processed.zip

echo "Now we can use the website UI demonstration system"
streamlit run demo.py