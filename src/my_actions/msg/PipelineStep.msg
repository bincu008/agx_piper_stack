string step_id
string function_name
# CDR-serialized request of the function's catalogued ROS service type.
uint8[] request_cdr
my_actions/StepBinding[] bindings
# stop or continue; failed results are never valid dependency inputs.
string failure_policy
