#from vocab import Vocabulary
import evaluation
import numpy

DATA_PATH = ''
RUN_PATH = 'models/'

shared_space = 'both' # help='both'|'object_text'|'activity_text' ;  default = 'both'

evaluation.evalrank(RUN_PATH+"msrvtt_object_text/model_best.pth.tar", RUN_PATH+"msrvtt_activity_text/model_best.pth.tar", data_path=DATA_PATH, split="test", shared_space="both")
