import os
from Classifier import Classifier
from target_encoder import get_target_encoder, transform_targets
from utils import read_csv_in_directory
from config import paths
from logger import get_logger, log_error
from schema.data_schema import load_json_data_schema, save_schema
from utils import set_seeds, ResourceTracker


logger = get_logger(task_name="train")


def run_training(
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
        train_dir: str = paths.TRAIN_DIR,
        predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
       ) -> None:
    """
       Run the training process and saves model artifacts

       Args:
           input_schema_dir (str, optional): The directory path of the input schema.
           saved_schema_dir_path (str, optional): The path where to save the schema.
           train_dir (str, optional): The directory path of the train data.
           predictor_dir_path (str, optional): Dir path where to save the predictor model.
       Returns:
           None
       """
    try:
        with ResourceTracker(logger, monitoring_interval=0.1):
            logger.info("Starting training...")
            set_seeds(seed_value=123)

            logger.info("Loading and saving schema...")
            data_schema = load_json_data_schema(input_schema_dir)
            save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

            logger.info("Loading training data...")
            x_train = read_csv_in_directory(train_dir)
            target_encoder = get_target_encoder(data_schema)
            transformed_targets = transform_targets(target_encoder, x_train)
            x_train[data_schema.target] = transformed_targets
            classifier = Classifier(x_train, data_schema)
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        classifier.save(predictor_dir_path)
        logger.info(f'The best model is: {type(classifier.model).__name__}')
        logger.info('Model saved!')

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
