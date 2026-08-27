Timeline file description:
1.msprof.json:Timeline report.
	1 CPU Layer: Data at the application layer, including the time consumption information of upper-layer application operators. The data needs to be collected only in msproftx or PyTorch scenarios.
	2 CANN Layer: Data at the CANN layer, including the time consumption data of components (such as AscendCL and Runtime) and nodes (operators).
	3 Ascend Hardware Layer: Bottom-layer NPU data, including the time consumption data and iteration trace data of each task stream under Ascend Hardware, Communication and Overlap Analysis communication data, and other system data.
	4 Overlap Analysis Layer: In cluster or multi-device scenarios, computation and communication are sometimes parallel. You can check the pipeline overlapping time (time when computation and communication are parallel) to determine the computation and communication efficiencies.
		Communication Layer: Communication time.
		Communication(Not Overlaopped) Layer: Communication time that is not overlapped.
		Computing: Computation time.
		Free: Interval.

Summary file description:
1.op_summary.csv:AI Core, AI CPU, AI Vector and COMMUNICATION communication operator data,is used to collect statistics on operator details and time consumptions.
	Op Name: Operator name.
	OP Type: Operator type. 
	Task Type: Task type. 
	Task Start Time: Task start time (μs).
	Task Duration: Task duration, including the scheduling time and the start time to the latest end time of the first core. The unit is μs.
	Task Wait Time: Interval between tasks (μs).(= this task's start_time - last task's start_time - last task's duration_time)
	Block Num: Number of running task blocks, which corresponds to the number of cores during task running.
	Mix Block Num: Number of running task blocks in Mix scenarios, which corresponds to the number of cores during task running.
	Context ID: Context ID.
	aiv_time: Average task execution duration on AIV.The value is calculated based on total_cycles and mix block num.
	aicore_time: Average task execution duration on AI Core.The value is calculated based on total_cycles and block num. The unit is μs. The data is inaccurate in the manual frequency modulation, dynamic frequency modulation (the power consumption exceeds the default value), and Atlas 300V/Atlas 300I Pro scenarios. You are not advised referring to it.
	total_cycles: Number of cycles taken to execute all task instructions.
	Register value: Value of the custom register whose data is to be collected.
2.task_time.csv:Task Scheduler summary.
	Waiting: Total wait time of a task (μs).
	Running: Total run time of a task (μs). An abnormally large value indicates that the operator implementation needs to be improved.
	Pending: Total pending time of a task (μs).
3.api_statistic.csv:Time spent by AscendCL API, is used to collect statistics on the time spent by API execution at the CANN layer.
	Level: Level of an API, including AscendCL, Runtime, Node, Model, and Communication.
4.op_statistic.csv:AI Core and AI CPU operator calling times and time consumption.
The parameters of the msprof command line tool are used as examples. The parameters of other collection modes are the same.Analyze the total calling time and total number of calls of each type of operators, check whether there are any operators with long total execution time, and analyze whether there is any optimization space for these operators.

