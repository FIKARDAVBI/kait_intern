��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_23/bias/v
z
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_23/kernel/v
�
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	d�*
dtype0
�
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_23/bias/m
z
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_23/kernel/m
�
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	d�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:�*
dtype0
|
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_23/kernel
u
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel* 
_output_shapes
:
��*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	d�*
dtype0
z
serving_default_input_8Placeholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8dense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_338872

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*

'	keras_api* 
.
0
1
2
3
%4
&5*
.
0
1
2
3
%4
&5*
* 
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 
�
5iter

6beta_1

7beta_2
	8decay
9learning_ratemUmVmWmX%mY&mZv[v\v]v^%v_&v`*

:serving_default* 

0
1*

0
1*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
'
0
1
2
3
4*

P0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
Q	variables
R	keras_api
	Stotal
	Tcount*

S0
T1*

Q	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vConst*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_339197
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_339282��
�	
�
D__inference_dense_22_layer_call_and_return_conditional_losses_338640

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
2__inference_embedding_model_7_layer_call_fn_338740
input_8
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338725p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
ڻ
�
__inference__traced_save_339197
file_prefix9
&read_disablecopyonread_dense_21_kernel:	d�5
&read_1_disablecopyonread_dense_21_bias:	�<
(read_2_disablecopyonread_dense_22_kernel:
��5
&read_3_disablecopyonread_dense_22_bias:	�<
(read_4_disablecopyonread_dense_23_kernel:
��5
&read_5_disablecopyonread_dense_23_bias:	�,
"read_6_disablecopyonread_adam_iter:	 .
$read_7_disablecopyonread_adam_beta_1: .
$read_8_disablecopyonread_adam_beta_2: -
#read_9_disablecopyonread_adam_decay: 6
,read_10_disablecopyonread_adam_learning_rate: )
read_11_disablecopyonread_total: )
read_12_disablecopyonread_count: C
0read_13_disablecopyonread_adam_dense_21_kernel_m:	d�=
.read_14_disablecopyonread_adam_dense_21_bias_m:	�D
0read_15_disablecopyonread_adam_dense_22_kernel_m:
��=
.read_16_disablecopyonread_adam_dense_22_bias_m:	�D
0read_17_disablecopyonread_adam_dense_23_kernel_m:
��=
.read_18_disablecopyonread_adam_dense_23_bias_m:	�C
0read_19_disablecopyonread_adam_dense_21_kernel_v:	d�=
.read_20_disablecopyonread_adam_dense_21_bias_v:	�D
0read_21_disablecopyonread_adam_dense_22_kernel_v:
��=
.read_22_disablecopyonread_adam_dense_22_bias_v:	�D
0read_23_disablecopyonread_adam_dense_23_kernel_v:
��=
.read_24_disablecopyonread_adam_dense_23_bias_v:	�
savev2_const
identity_51��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_21_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	d�z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_21_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_22_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_22_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_23_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_23_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_adam_iter^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_adam_beta_1^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_adam_beta_2^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_adam_decay^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead,read_10_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp,read_10_disablecopyonread_adam_learning_rate^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_11/DisableCopyOnReadDisableCopyOnReadread_11_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpread_11_disablecopyonread_total^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_12/DisableCopyOnReadDisableCopyOnReadread_12_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpread_12_disablecopyonread_count^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_dense_21_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_dense_21_kernel_m^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_adam_dense_21_bias_m"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_adam_dense_21_bias_m^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_dense_22_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_dense_22_kernel_m^Read_15/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_dense_22_bias_m"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_dense_22_bias_m^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_dense_23_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_dense_23_kernel_m^Read_17/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_dense_23_bias_m"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_dense_23_bias_m^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_dense_21_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_dense_21_kernel_v^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_dense_21_bias_v"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_dense_21_bias_v^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_adam_dense_22_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_adam_dense_22_kernel_v^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_dense_22_bias_v"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_dense_22_bias_v^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_dense_23_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_dense_23_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_dense_23_bias_v"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_dense_23_bias_v^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *(
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_50Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_51IdentityIdentity_50:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_51Identity_51:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
D__inference_dense_23_layer_call_and_return_conditional_losses_339024

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_338986

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338725

inputs"
dense_21_338702:	d�
dense_21_338704:	�#
dense_22_338707:
��
dense_22_338709:	�#
dense_23_338712:
��
dense_23_338714:	�
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_338702dense_21_338704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_338624�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_338707dense_22_338709*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_338640�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_338712dense_23_338714*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_338656�
*tf.math.l2_normalize_7/l2_normalize/SquareSquare)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMul)dense_23/StatefulPartitionedCall:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
2__inference_embedding_model_7_layer_call_fn_338906

inputs
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338768p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�m
�
"__inference__traced_restore_339282
file_prefix3
 assignvariableop_dense_21_kernel:	d�/
 assignvariableop_1_dense_21_bias:	�6
"assignvariableop_2_dense_22_kernel:
��/
 assignvariableop_3_dense_22_bias:	�6
"assignvariableop_4_dense_23_kernel:
��/
 assignvariableop_5_dense_23_bias:	�&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: =
*assignvariableop_13_adam_dense_21_kernel_m:	d�7
(assignvariableop_14_adam_dense_21_bias_m:	�>
*assignvariableop_15_adam_dense_22_kernel_m:
��7
(assignvariableop_16_adam_dense_22_bias_m:	�>
*assignvariableop_17_adam_dense_23_kernel_m:
��7
(assignvariableop_18_adam_dense_23_bias_m:	�=
*assignvariableop_19_adam_dense_21_kernel_v:	d�7
(assignvariableop_20_adam_dense_21_bias_v:	�>
*assignvariableop_21_adam_dense_22_kernel_v:
��7
(assignvariableop_22_adam_dense_22_bias_v:	�>
*assignvariableop_23_adam_dense_23_kernel_v:
��7
(assignvariableop_24_adam_dense_23_bias_v:	�
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_22_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_22_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_23_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_23_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_21_kernel_mIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_21_bias_mIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_22_kernel_mIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_22_bias_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_23_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_23_bias_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_21_kernel_vIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_21_bias_vIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_22_kernel_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_22_bias_vIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_23_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_23_bias_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
D__inference_dense_23_layer_call_and_return_conditional_losses_338656

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338670
input_8"
dense_21_338625:	d�
dense_21_338627:	�#
dense_22_338641:
��
dense_22_338643:	�#
dense_23_338657:
��
dense_23_338659:	�
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_21_338625dense_21_338627*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_338624�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_338641dense_22_338643*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_338640�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_338657dense_23_338659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_338656�
*tf.math.l2_normalize_7/l2_normalize/SquareSquare)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMul)dense_23/StatefulPartitionedCall:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
�
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338696
input_8"
dense_21_338673:	d�
dense_21_338675:	�#
dense_22_338678:
��
dense_22_338680:	�#
dense_23_338683:
��
dense_23_338685:	�
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_21_338673dense_21_338675*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_338624�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_338678dense_22_338680*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_338640�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_338683dense_23_338685*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_338656�
*tf.math.l2_normalize_7/l2_normalize/SquareSquare)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMul)dense_23/StatefulPartitionedCall:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
�.
�
!__inference__wrapped_model_338609
input_8L
9embedding_model_7_dense_21_matmul_readvariableop_resource:	d�I
:embedding_model_7_dense_21_biasadd_readvariableop_resource:	�M
9embedding_model_7_dense_22_matmul_readvariableop_resource:
��I
:embedding_model_7_dense_22_biasadd_readvariableop_resource:	�M
9embedding_model_7_dense_23_matmul_readvariableop_resource:
��I
:embedding_model_7_dense_23_biasadd_readvariableop_resource:	�
identity��1embedding_model_7/dense_21/BiasAdd/ReadVariableOp�0embedding_model_7/dense_21/MatMul/ReadVariableOp�1embedding_model_7/dense_22/BiasAdd/ReadVariableOp�0embedding_model_7/dense_22/MatMul/ReadVariableOp�1embedding_model_7/dense_23/BiasAdd/ReadVariableOp�0embedding_model_7/dense_23/MatMul/ReadVariableOp�
0embedding_model_7/dense_21/MatMul/ReadVariableOpReadVariableOp9embedding_model_7_dense_21_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
!embedding_model_7/dense_21/MatMulMatMulinput_88embedding_model_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1embedding_model_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp:embedding_model_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"embedding_model_7/dense_21/BiasAddBiasAdd+embedding_model_7/dense_21/MatMul:product:09embedding_model_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
embedding_model_7/dense_21/ReluRelu+embedding_model_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0embedding_model_7/dense_22/MatMul/ReadVariableOpReadVariableOp9embedding_model_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!embedding_model_7/dense_22/MatMulMatMul-embedding_model_7/dense_21/Relu:activations:08embedding_model_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1embedding_model_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp:embedding_model_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"embedding_model_7/dense_22/BiasAddBiasAdd+embedding_model_7/dense_22/MatMul:product:09embedding_model_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0embedding_model_7/dense_23/MatMul/ReadVariableOpReadVariableOp9embedding_model_7_dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!embedding_model_7/dense_23/MatMulMatMul+embedding_model_7/dense_22/BiasAdd:output:08embedding_model_7/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1embedding_model_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp:embedding_model_7_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"embedding_model_7/dense_23/BiasAddBiasAdd+embedding_model_7/dense_23/MatMul:product:09embedding_model_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<embedding_model_7/tf.math.l2_normalize_7/l2_normalize/SquareSquare+embedding_model_7/dense_23/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Kembedding_model_7/tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
9embedding_model_7/tf.math.l2_normalize_7/l2_normalize/SumSum@embedding_model_7/tf.math.l2_normalize_7/l2_normalize/Square:y:0Tembedding_model_7/tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(�
?embedding_model_7/tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
=embedding_model_7/tf.math.l2_normalize_7/l2_normalize/MaximumMaximumBembedding_model_7/tf.math.l2_normalize_7/l2_normalize/Sum:output:0Hembedding_model_7/tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
;embedding_model_7/tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrtAembedding_model_7/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
5embedding_model_7/tf.math.l2_normalize_7/l2_normalizeMul+embedding_model_7/dense_23/BiasAdd:output:0?embedding_model_7/tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:�����������
IdentityIdentity9embedding_model_7/tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp2^embedding_model_7/dense_21/BiasAdd/ReadVariableOp1^embedding_model_7/dense_21/MatMul/ReadVariableOp2^embedding_model_7/dense_22/BiasAdd/ReadVariableOp1^embedding_model_7/dense_22/MatMul/ReadVariableOp2^embedding_model_7/dense_23/BiasAdd/ReadVariableOp1^embedding_model_7/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2f
1embedding_model_7/dense_21/BiasAdd/ReadVariableOp1embedding_model_7/dense_21/BiasAdd/ReadVariableOp2d
0embedding_model_7/dense_21/MatMul/ReadVariableOp0embedding_model_7/dense_21/MatMul/ReadVariableOp2f
1embedding_model_7/dense_22/BiasAdd/ReadVariableOp1embedding_model_7/dense_22/BiasAdd/ReadVariableOp2d
0embedding_model_7/dense_22/MatMul/ReadVariableOp0embedding_model_7/dense_22/MatMul/ReadVariableOp2f
1embedding_model_7/dense_23/BiasAdd/ReadVariableOp1embedding_model_7/dense_23/BiasAdd/ReadVariableOp2d
0embedding_model_7/dense_23/MatMul/ReadVariableOp0embedding_model_7/dense_23/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
�	
�
2__inference_embedding_model_7_layer_call_fn_338783
input_8
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338768p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_338624

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338768

inputs"
dense_21_338745:	d�
dense_21_338747:	�#
dense_22_338750:
��
dense_22_338752:	�#
dense_23_338755:
��
dense_23_338757:	�
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_338745dense_21_338747*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_338624�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_338750dense_22_338752*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_338640�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_338755dense_23_338757*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_338656�
*tf.math.l2_normalize_7/l2_normalize/SquareSquare)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMul)dense_23/StatefulPartitionedCall:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�$
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338966

inputs:
'dense_21_matmul_readvariableop_resource:	d�7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0|
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMuldense_22/BiasAdd:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*tf.math.l2_normalize_7/l2_normalize/SquareSquaredense_23/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMuldense_23/BiasAdd:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_23_layer_call_fn_339014

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_338656p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_22_layer_call_fn_338995

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_338640p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
2__inference_embedding_model_7_layer_call_fn_338889

inputs
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338725p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_338872
input_8
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_338609p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_8
�$
�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338936

inputs:
'dense_21_matmul_readvariableop_resource:	d�7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�;
'dense_23_matmul_readvariableop_resource:
��7
(dense_23_biasadd_readvariableop_resource:	�
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0|
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_23/MatMulMatMuldense_22/BiasAdd:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*tf.math.l2_normalize_7/l2_normalize/SquareSquaredense_23/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9tf.math.l2_normalize_7/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
'tf.math.l2_normalize_7/l2_normalize/SumSum.tf.math.l2_normalize_7/l2_normalize/Square:y:0Btf.math.l2_normalize_7/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_7/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_7/l2_normalize/MaximumMaximum0tf.math.l2_normalize_7/l2_normalize/Sum:output:06tf.math.l2_normalize_7/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_7/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_7/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_7/l2_normalizeMuldense_23/BiasAdd:output:0-tf.math.l2_normalize_7/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������w
IdentityIdentity'tf.math.l2_normalize_7/l2_normalize:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������d: : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_21_layer_call_fn_338975

inputs
unknown:	d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_338624p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
D__inference_dense_22_layer_call_and_return_conditional_losses_339005

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_80
serving_default_input_8:0���������dK
tf.math.l2_normalize_71
StatefulPartitionedCall:0����������tensorflow/serving/predict:�m
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
J
0
1
2
3
%4
&5"
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
-trace_0
.trace_1
/trace_2
0trace_32�
2__inference_embedding_model_7_layer_call_fn_338740
2__inference_embedding_model_7_layer_call_fn_338783
2__inference_embedding_model_7_layer_call_fn_338889
2__inference_embedding_model_7_layer_call_fn_338906�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0z.trace_1z/trace_2z0trace_3
�
1trace_0
2trace_1
3trace_2
4trace_32�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338670
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338696
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338936
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338966�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1z3trace_2z4trace_3
�B�
!__inference__wrapped_model_338609input_8"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
5iter

6beta_1

7beta_2
	8decay
9learning_ratemUmVmWmX%mY&mZv[v\v]v^%v_&v`"
	optimizer
,
:serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
)__inference_dense_21_layer_call_fn_338975�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
�
Atrace_02�
D__inference_dense_21_layer_call_and_return_conditional_losses_338986�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
": 	d�2dense_21/kernel
:�2dense_21/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
)__inference_dense_22_layer_call_fn_338995�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
D__inference_dense_22_layer_call_and_return_conditional_losses_339005�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
#:!
��2dense_22/kernel
:�2dense_22/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
)__inference_dense_23_layer_call_fn_339014�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
D__inference_dense_23_layer_call_and_return_conditional_losses_339024�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
#:!
��2dense_23/kernel
:�2dense_23/bias
"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_embedding_model_7_layer_call_fn_338740input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_embedding_model_7_layer_call_fn_338783input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_embedding_model_7_layer_call_fn_338889inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_embedding_model_7_layer_call_fn_338906inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338670input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338696input_8"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338936inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338966inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_338872input_8"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_21_layer_call_fn_338975inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_21_layer_call_and_return_conditional_losses_338986inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_22_layer_call_fn_338995inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_22_layer_call_and_return_conditional_losses_339005inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_23_layer_call_fn_339014inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_23_layer_call_and_return_conditional_losses_339024inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
Q	variables
R	keras_api
	Stotal
	Tcount"
_tf_keras_metric
.
S0
T1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
':%	d�2Adam/dense_21/kernel/m
!:�2Adam/dense_21/bias/m
(:&
��2Adam/dense_22/kernel/m
!:�2Adam/dense_22/bias/m
(:&
��2Adam/dense_23/kernel/m
!:�2Adam/dense_23/bias/m
':%	d�2Adam/dense_21/kernel/v
!:�2Adam/dense_21/bias/v
(:&
��2Adam/dense_22/kernel/v
!:�2Adam/dense_22/bias/v
(:&
��2Adam/dense_23/kernel/v
!:�2Adam/dense_23/bias/v�
!__inference__wrapped_model_338609�%&0�-
&�#
!�
input_8���������d
� "P�M
K
tf.math.l2_normalize_71�.
tf_math_l2_normalize_7�����������
D__inference_dense_21_layer_call_and_return_conditional_losses_338986d/�,
%�"
 �
inputs���������d
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_21_layer_call_fn_338975Y/�,
%�"
 �
inputs���������d
� ""�
unknown�����������
D__inference_dense_22_layer_call_and_return_conditional_losses_339005e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_22_layer_call_fn_338995Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_23_layer_call_and_return_conditional_losses_339024e%&0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_23_layer_call_fn_339014Z%&0�-
&�#
!�
inputs����������
� ""�
unknown�����������
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338670q%&8�5
.�+
!�
input_8���������d
p

 
� "-�*
#� 
tensor_0����������
� �
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338696q%&8�5
.�+
!�
input_8���������d
p 

 
� "-�*
#� 
tensor_0����������
� �
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338936p%&7�4
-�*
 �
inputs���������d
p

 
� "-�*
#� 
tensor_0����������
� �
M__inference_embedding_model_7_layer_call_and_return_conditional_losses_338966p%&7�4
-�*
 �
inputs���������d
p 

 
� "-�*
#� 
tensor_0����������
� �
2__inference_embedding_model_7_layer_call_fn_338740f%&8�5
.�+
!�
input_8���������d
p

 
� ""�
unknown�����������
2__inference_embedding_model_7_layer_call_fn_338783f%&8�5
.�+
!�
input_8���������d
p 

 
� ""�
unknown�����������
2__inference_embedding_model_7_layer_call_fn_338889e%&7�4
-�*
 �
inputs���������d
p

 
� ""�
unknown�����������
2__inference_embedding_model_7_layer_call_fn_338906e%&7�4
-�*
 �
inputs���������d
p 

 
� ""�
unknown�����������
$__inference_signature_wrapper_338872�%&;�8
� 
1�.
,
input_8!�
input_8���������d"P�M
K
tf.math.l2_normalize_71�.
tf_math_l2_normalize_7����������