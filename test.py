import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit
from moabb.evaluations import AllRunsEvaluation
from shallow import CollapsedShallowNet

from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.utils import setup_seed

mne.set_log_level(False)

# Print Information PyTorch
print(f"Torch Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

seed = 42
setup_seed(seed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameter
LEARNING_RATE = 0.0625 * 0.01  # parameter taken from Braindecode
WEIGHT_DECAY = 0  # parameter taken from Braindecode
BATCH_SIZE = 64  # parameter taken from BrainDecode
EPOCH = 10
PATIENCE = 3
fmin = 4
fmax = 100
tmin = 0
tmax = None


dataset = BNCI2014_001()
events = ["right_hand", "left_hand"]
paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
)
subjects = [1]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)

# Define the CollapsedShallowNet model within the EEGClassifier
clf = EEGClassifier(
    module=CollapsedShallowNet,  # Use your model here
    module__n_chans=X.shape[1],  # number of input channels
    module__n_outputs=len(events),  # number of output classes
    module__n_times=X.shape[2],  # length of the input signal in time points
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=seed),
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=1,
)

# Create a pipeline with the classifier
pipes = {"CollapsedShallowNet": make_pipeline(clf)}

# Use the same CrossSessionEvaluation for training and evaluation
evaluation = AllRunsEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    suffix="braindecode_collapsedshallownet_example",
    overwrite=True,
    return_epochs=True,
    n_jobs=1,
)

# Run the evaluation process
results = evaluation.process(pipes)

# Display the results
print(results.head())
