# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('darkdetect')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['e:\\Work\\Vscode\\social_media\\watermark_remover_gui.py'],
    pathex=['D:\\apps\\anaconda\\Library\\bin', 'D:\\apps\\anaconda\\DLLs'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pytest', 'sphinx', 'IPython', 'notebook', 'jupyter', 'numba',
              'pandas', 'pyarrow', 'sqlalchemy', 'openpyxl', 'tables', 'botocore',
              's3fs', 'streamlit', 'gensim', 'moviepy', 'torch', 'torchvision',
              'simple_lama_inpainting', 'scipy', 'skimage', 'nltk', 'statsmodels',
              'plotly', 'panel', 'pyviz_comms', 'lxml', 'cryptography', 'bcrypt',
              'nacl', 'zmq', 'cloudpickle'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WatermarkRemover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
