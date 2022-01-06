import datetime

COVID_CUT_OFF = datetime.datetime.strptime("3/12/20 00:00","%m/%d/%y %H:%M") 
SURVEY_WINDOW_DAYS = 14

LOCKDOWN_YELLOW_START = datetime.datetime.strptime("5/15/20 00:00","%m/%d/%y %H:%M") 
LOCKDOWN_GREEN_START = datetime.datetime.strptime("6/5/20 00:00","%m/%d/%y %H:%M") 