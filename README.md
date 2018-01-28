Code for [Evolution Strategy for playing Catch](https://edersantana.github.io/articles/keras_rl/) blog post

### Train
```bash
python evolve.py
```

### Test
1) Generate figures
```bash
python test.py
```

2) Make gif
```bash
ffmpeg -i %04d.png output.gif -vf fps=1
```

### Requirements
* Prior supervised learning and Keras knowledge
* Python science stack (numpy, scipy, matplotlib) - Install Anaconda!
* Tensorflow (last test on version 1.4)
* Keras (last testest on commit b0303f03ff03)
* ffmpeg (optional)

### License
This code is released under [MIT license](https://en.wikipedia.org/wiki/MIT_License). 
