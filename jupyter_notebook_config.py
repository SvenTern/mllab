# Jupyter Notebook configuration file

# Allow notebook to be accessed from any IP address
c.NotebookApp.ip = '0.0.0.0'

# Use port 8888 for the notebook server
c.NotebookApp.port = 8888

# Do not open a browser automatically
c.NotebookApp.open_browser = False

# Allow root user to run the notebook
c.NotebookApp.allow_root = True

# Set a password for accessing the notebook
from notebook.auth import passwd
c.NotebookApp.password = passwd("my001svs")

# Disable authentication token (not recommended unless password is set)
c.NotebookApp.token = ''

# Enable CORS (Cross-Origin Resource Sharing) if needed
c.NotebookApp.allow_origin = '*'

# Set the notebook server to start in a specific directory (optional)
# c.NotebookApp.notebook_dir = '/jupyter'

# Additional logging for debugging
c.Application.log_level = 'INFO'