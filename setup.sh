mkdir -p ~/.streamlit/
echo "
[general]
email = \"benoit.rouffet@gmail.com\"
" > ~/.streamlit/credentials.toml
echo " 
[serveur]
headless = true
port = $PORT 
enableCORS = false 
" > ~/.streamlit/config.toml