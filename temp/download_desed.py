import desed
desed.download_real("./data/dataset")
desed.download_desed_soundbank("./data/soundbank")
# Additional sets:
desed.download_fuss("./data/FUSS")
desed.download_fsd50k("./data/fsd50k", gtruth_only=True) # groundtruth only to use annotations for FUSS