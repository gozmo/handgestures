find dataset/labeled/metal/ | sort -r | head -n 30 | xargs -i mv {} dataset/holdout/metal/
find dataset/labeled/none/ | sort -r | head -n 30 | xargs -i mv {} dataset/holdout/none/
find dataset/labeled/ok/ | sort -r | head -n 30 | xargs -i mv {} dataset/holdout/ok/
find dataset/labeled/victory/ | sort -r | head -n 30 | xargs -i mv {} dataset/holdout/victory/
