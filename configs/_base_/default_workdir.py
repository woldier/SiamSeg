# -*- coding:utf-8 -*-
from datetime import datetime
now = datetime.now()
current_date = now.strftime('%Y-%m-%d')
current_time = now.strftime('%H-%M-%S')
work_dir = rf'./result/{current_date}/{current_time}_{{fileBasenameNoExtension}}'