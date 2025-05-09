# Goal Opportunity Spotting (Master's Thesis)

Reg. No.: FIIT-182905-102924

This project focuses on detecting goal opportunities in football matches.

### Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/RomanGrom/goal-opportunity-spotting-dp.git
   cd goal-opportunity-spotting-dp

2. In VS Code just Reopen in container. In console:

   ```bash
   ./docker/build.sh
   ./docker/run.sh 0


### Reproducing results

1. The models are stored on the school NFS at `/mnt/nfs-data/public/xgromr/models`, and the dataset is stored as `chances_dataset.tar` on `/mnt/nfs-data/public/xgromr`. You need to untar the dataset first:

   ```bash
   tar -xvf chances_dataset.tar

2. After extracting the dataset, use the project/actions_to_csv.py script, which processes the entire dataset with the model and stores the values. Make sure to set the correct paths to the dataset, model, and output file at the beginning of the script.

3. The next step is to use the utils/evaluate_csv.py script. Set the path to the .csv file correctly, and it will print out all the relevant metrics.

4. Alternatively, you can use the Jupyter notebook notebooks/pekne_grafy.ipynb to visualize the results with graphs.


### Models

1. **classic_actor.zip** - This model comes from the "Classic actor-critic approach" experiment.

2. **onlyai_3.zip** - This model is from the "Only AI critic approach" experiment. The results in the document were produced using a model that is no longer available, which is about 1% less successful than this one. The `.csv` file from the model described in the document can be found in `data/onlyai_best.csv`.

3. **onlyai_4.zip** - This model's results are presented in the "Special cases" section. For these special cases, you need to uncomment the corresponding line in the script.

4. **onlyai_6.zip** - This model is from the "Only AI critic with lower discount factor" chapter.


### Other Files

1. **project/main.py** - Script for running the training process.
2. **project/trainer.py** - Defines the hyperparameters and training setup.
3. **project/custom_networks.py** - Contains definitions for custom models.
4. **project/custom_callback.py** - Custom callback used during training, e.g., for logging.
5. **project/football_gym.py** - Defines the football environment for simulation.

6. **project/actions_to_csv.py** - Script to iterate through the dataset and save values to a CSV file.
7. **project/evaluate_gfootball.py** - Program to test if the critic can recognize goals in the simulator.
8. **project/watch_agent_play.py** - Program to save videos of the agent playing in the simulation.

9. **utils/** - Contains files for downloading SoccerNet, saving frames, and similar tasks.
10. **notebooks/** - Contains Jupyter notebooks for analyzing and visualizing the experiments.













<<<<<<< HEAD
Reg. No.: FIIT-182905-102924
=======
>>>>>>> 93b96fabd187eb522b3bc9245dcc342585893425

