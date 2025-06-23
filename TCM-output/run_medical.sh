export CUDA_VISIBLE_DEVICES=6,8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PATH="/home/gpuall/.conda/envs/zwh_vllm/bin:$PATH"

model_names=(
    # 'Baichuan2-7B-Chat'
    # 'Baichuan2-13B-Chat'
    # 'Qwen2.5-0.5B-Instruct'
    # 'Qwen2.5-3B-Instruct'
    # 'Qwen1.5-7B-Chat'
    # 'Qwen1.5-14B-Chat'
    # 'Qwen2.5-7B-Instruct'
    # 'Qwen2.5-14B-Instruct'
    # 'glm-4-9b-chat'
    # 'Qwen2.5-7B'
    # 'Qwen1.5-14B'
    # 'Qwen1.5-7B'
    # 'Baichuan2-13B-Base'
    # 'Baichuan2-7B-Base'
    # 'glm-4-9b'
    # 'DeepSeek-R1-Distill-Qwen-7B'
    # 'DeepSeek-R1-Distill-Qwen-14B'
    # 'Internlm2.5-7B'
    # 'internlm2_5-7b-chat'
    # 'internlm3-8b-instruct'
    # 'Qwen2.5-7B-Instruct-poem-GRPO-v1'
    # 'Qwen2.5-7B-Instruct-gsm-GRPO-baseline'
    # 'Qwen2.5-32B-Instruct-poem-GRPO-v1'
    # 'Qwen2.5-32B-Instruct'
    # 'Qwen2.5-32B-Instruct-poem-GRPO-v2'
    # 'Qwen2.5-32B-Instruct-poem-GRPO-v3'
    # 'Qwen2.5-32B-Instruct-poem-GRPO-v3.1'
    # 'QwQ-32B'
    # 'GLM-4-Z1-32B-0414'
    'Qwen3-14B'
)

# Model path mapping
declare -A model_path_map=(
    ['Qwen2.5-7B-Instruct-poem-GRPO-v2']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-7B-Instruct-poem-GRPO-v2/checkpoint-80'
    ['Qwen2.5-7B-Instruct-poem-GRPO-v1']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-7B-Instruct-poem-GRPO-v1/checkpoint-80'
    ['Qwen2.5-32B-Instruct-poem-GRPO-v1']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-32B-Instruct-poem-GRPO-v1/checkpoint-180'
    ['Qwen2.5-32B-Instruct-poem-GRPO-v2']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-32B-Instruct-poem-GRPO-v2/checkpoint-180'
    ['Qwen2.5-32B-Instruct-poem-GRPO-v3']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-32B-Instruct-poem-GRPO-v3/checkpoint-180'
    ['Qwen2.5-32B-Instruct-poem-GRPO-v3.1']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-32B-Instruct-poem-GRPO-v3/checkpoint-180'
    ['Qwen2.5-7B-Instruct-gsm-GRPO-baseline']='/home/gpuall/model_output/zwh/grpo/model_output/Qwen2.5-32B-Instruct-poem-GRPO-v3/checkpoint-180'
)

# 遍历每个模型名称
for model_name in "${model_names[@]}"; do
    # 根据模型名称设置模板
    case $model_name in
       "Internlm2.5-7B" | "internlm2_5-7b-chat" | "internlm3-8b-instruct")
            template=intern2
            lora_target="all"
            ;;
        "Qwen3-14B")
            template=qwen
            lora_target="all"
            ;;
        "DeepSeek-R1-Distill-Qwen-7B" | "DeepSeek-R1-Distill-Qwen-14B")
            template=deepseek3
            lora_target="all"
            ;;
        "Xunzi-Qwen1.5-7B" | "Xunzi-Qwen1.5-14B" | "Xunzi-Qwen-7B" | "Xunzi-Qwen1.5-7B_chat" | "Xunzi-Qwen2-7B")
            template=qwen
            lora_target="all"
            ;;
        "Xunzi-Baichuan2-7B" | "Baichuan2-7B-Base" | "Baichuan2-7B-Chat" | "Baichuan2-13B-Chat" | "Baichuan2-13B-Base")
            template=baichuan2
            lora_target=all
            ;;
        "chatglm3-6b")
            template=chatglm3
            lora_target="all"
            ;;
        "Llama2-7B")
            template=llama2  # 使用相同的模板
            lora_target="all"
            ;;
        "Llama3-8B")
            template=llama3  # 使用相同的模板
            lora_target="all"
            ;;
        "QwQ-32"| "Qwen1.5-7B" | "Qwen1.5-14B" | "Qwen2-1.5B" | "Qwen2-7B" | "Qwen1.5-7B-Chat" | "Qwen1.5-14B-Chat" | "Qwen2-1.5B-Instruct" | "Qwen2-7B-Instruct" | "Qwen2.5-7B-Instruct" | "Qwen2.5-0.5B-Instruct" | "Qwen2.5-3B-Instruct" | "Qwen2.5-14B-Instruct")
            template=qwen
            lora_target="all"
            ;;
        "Qwen2.5-32B-Instruct-poem-GRPO-v3.1" | "Qwen2.5-7B-Instruct-poem-GRPO-v1" | "Qwen2.5-7B-Instruct-poem-GRPO-v2" | "Qwen2.5-32B-Instruct-poem-GRPO-v1" | "Qwen2.5-32B-Instruct-poem-GRPO-v2" | "Qwen2.5-32B-Instruct" | "Qwen2.5-32B-Instruct-poem-GRPO-v3")
            template=poem_grpo_qwen
            lora_target="all"
            ;;
        "Xunzi-Glm3-6B")
            template=chatglm3
            lora_target="all"
            ;;
        "glm-4-9b-chat" | "Xunzi-glm4-9b" | "glm-4-9b")
            template=glm4
            lora_target="all"
            ;;
        *)
            echo "Unknown model name: $model_name"
            continue
            ;;
    esac

    # 定义数据集和训练数据列表
    datasets=(
        'SFT_medicalKnowledge_test_1000'
        # 'test_noreasoning'
    )

    train_data=(
        'medical_train'
        # 'poemsc_train_noreasoning'
    )

    # 遍历每个数据集
    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset="${datasets[i]}"
        train_dataset="${train_data[i]}"
        data_path=./data/SFT/${dataset}.json
        output_path=./output/SFT_lora/$dataset/$model_name
        
        # Use the mapped path if available, otherwise use the default path
        if [[ -n "${model_path_map[$model_name]}" ]]; then
            model_path="${model_path_map[$model_name]}"
        else
            model_path=/home/gpuall/ifs_data/pre_llms/$model_name
        fi
        # lora_path 这个是合并lora权重的，我发你的是完整模型，直接注释这个，改成完整路径就好
        lora_path="/home/gpuall/model_output/zwh/medical/$train_dataset/$model_name"
        log_path=log/SFT_lora/$model_name/$dataset
        mkdir -p $log_path
        
        # 输出当前处理的信息
        echo "Processing model: $model_name, dataset: $dataset"
        echo "Model path: $model_path"
        echo "Data path: $data_path"
        echo "Output path: $output_path"
        echo "Log path: $log_path"

        # 运行预测脚本
        python predict.py \
            --model_path $model_path \
            --max_tokens 1024 \
            --template $template \
            --data_path $data_path \
            --output_path $output_path \
            --gpu_memory_utilization 0.9 \
            --gpu_num 2 \
            --temperature 0.6 \
            --lora_path $lora_path \
            --max_len 1024 > $log_path/$model_name.log 2>&1 & \
        # 等待任务完成
        wait
    done
done