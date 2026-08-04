### this is to launch phy on Mac M1 with OpenGL 2.1 and QtWebEngine

import os
from pathlib import Path

from PyQt5.QtGui import QSurfaceFormat

from phy.apps.template import template_gui

# Set environment variables BEFORE importing QtWebEngine
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = (
    '--enable-gpu --use-gl=desktop --enable-features=VaapiVideoDecoder'
)
os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'


# Configure OpenGL for both PyQt5 and QtWebEngine
def setup_opengl_for_webengine():
    # This must be called before QApplication creation
    # QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    format = QSurfaceFormat()
    format.setVersion(2, 1)  # Mac M1 maximum
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    format.setSamples(4)
    format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)

    QSurfaceFormat.setDefaultFormat(format)


# Apply configuration
setup_opengl_for_webengine()
# Now all OpenGL contexts created will use this format
# %%
HOME_PATH = Path().home()
BASE_PATH = HOME_PATH / 'GLC7189_c_g0'
PARAMS_PATH = BASE_PATH / 'sorting' / 'kilosort4_nblock1_raventest' / 'sorter_output' / 'params.py'

NEW_DAT_PATH = BASE_PATH / 'preprocessed' / 'traces_cached_seg0.raw'

# Update dat_path in params.py
with open(PARAMS_PATH, "r") as f:
    lines = f.readlines()

with open(PARAMS_PATH, "w") as f:
    for line in lines:
        if line.strip().startswith("dat_path"):
            f.write(f"dat_path = r'{NEW_DAT_PATH}'\n")
        else:
            f.write(line)

# %%
template_gui(PARAMS_PATH)
