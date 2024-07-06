block_cipher = None

a = Analysis(['Vincent Kane Skateboard Parts Recommender.py'],
             pathex=[r'C:\Users\vkane\Documents\Eastern University\DTSC450\Project\Final Files\Skateboard Parts Recommender'],
             binaries=[],
             datas=[
                 ('decks.csv', '.'), 
                 ('wheels.csv', '.'), 
                 ('trucks.csv', '.'), 
                 ('skateboard.ico', '.'), 
                 ('skateboard_image.png', '.')
             ],
             hiddenimports=[
                'pandas', 'sklearn', 'numpy', 'PIL', 'tkinter'
            ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Skateboard Parts Recommender',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,  # Set to True if you want a console window
          icon='skateboard.ico')  # Use the icon file if available
