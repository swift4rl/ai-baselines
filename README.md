# ai-baselines 

Swift4RL is an open source organization that explores new way of working with Reinforcement Learning combining [Swift for TensorFlow (S4TF)](https://www.tensorflow.org/swift), [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) and possibly other environments in the future. Python has been the main programming language 
for data scientist for so long eventhough it has some drawbacks such as being slow, concurrency (GIL), and relying on native bindings for wiring Python with C++ and CUDA for resource intensive applications. On the other hand [Swift for Tensorflow](https://www.tensorflow.org/swift) provides easy way to write custom ops, debugging, aesthetics and design which allows power users like AI researches to become productive.  Another advantage of Swift is that it's been built for mobile devices. Therefore it is pretty lightweight. Wouldn't it be awesome to train baseline models by taking advantage of Unity's physic engine and deploy them on the real world mobile robots.


## Requirements

* Xcode 12
* macOS 10.15.6 and above
* Swift 5.3
* Unity Hub
* protoc-4.0.0-rc-2
* grpc_csharp_plugin

## Installation

Install rake and xcodeproj as follows and then execute rake dependencies that will generate xcode project with deployment target
macOS 10.15

```bash
[sudo] gem install rake
[sudo] gem install xcodeproj
rake dependencies
```

# DS Experiments
To run with python gym environments one must have a the python packages installed. Swift must also know where python is where the dependencies are installed.

## On OSX

### Install python 3.8
Install python 3.8 as a system python version, not a virtual env. We could not get that to work, e.g. in a venv or anaconda env.

```bash
brew install python@3.8
```

### Install python dependencies
In your newly installed python 3.8 install dependencies. The path is on my mac "/usr/local/Frameworks/Python.framework/Versions/bin/python3"

Use your path as from the example above

```bash
PATH/python3 -m pip gym numpy matplotlib mlagents tensorflow tensorboard
```

If you have not installed pip in the python version above do it as in [here](https://pip.pypa.io/en/stable/installing/). With the python path as above.

### Run with environment variable
Swift makes it possible to choose which python version to use. Documentation is [here](https://www.tensorflow.org/swift/tutorials/python_interoperability), but not the easiest to get it to work. We could not get it to work with a venv or anaconda env.

You will need to change the path to the environment inside Run/main.swift to an executable environment. Docs to create an executable environment [link](https://github.com/Unity-Technologies/ml-agents/blob/release_7_docs/docs/Learning-Environment-Executable.md)
```bash
PYTHON_LIBRARY=PATH/python3 swift run
```

Then the model should run and train a PPO model on the environment. This also saves logs to tensorboard via the python api. To run the tensorboard with correct logdir run:

```bash
tensorboard --logdir=logs
```


