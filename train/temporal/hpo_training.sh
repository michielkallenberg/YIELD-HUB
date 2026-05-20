#!/bin/bash
#SBATCH --job-name=cybench_final_training
#SBATCH --time=48:00:00
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cybench
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Generate unique run ID (timestamp-based for traceability)
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
echo "Run ID: $RUN_ID"

MIN_YEARS=8
EXCLUDED_COUNTRIES=""

YEARS_DICT=$(cat <<'EOF'
{
'maize':
    {'AO': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'AR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'AT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BE': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BF': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019],
    'BG': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'CN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CZ': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'DE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'DK': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EE': [],
    'EL': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'ES': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'ET': [2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2020],
    'FI': [],
    'FR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HR': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HU': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'IE': [],
    'IN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'IT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LS': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'LT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LV': [],
    'MG': [2005, 2006, 2007, 2008, 2009, 2010],
    'ML': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'MW': [2018, 2019, 2020, 2021, 2022, 2023],
    'MX': [2014, 2017, 2019, 2022],
    'MZ': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'NE': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2020, 2021],
    'NL': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PL': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'RO': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SE': [2007, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    'SK': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'SN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    'TD': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'US': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'ZA': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'ZM': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]},
'wheat': {
    'AR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'AT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'AU': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'BE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BG': [2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    'BR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CN': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CZ': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'DE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'DK': [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'ES': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'FI': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'FR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HR': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HU': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'IE': [2010, 2011, 2012, 2019, 2020],
    'IN': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'IT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LV': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'NL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'RO': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SK': [2017, 2018],
    'US': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]}}
EOF
)

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import sys
import os

# Suppress module-level debug output
sys.stdout = open(os.devnull, 'w')
import ast

# Restore stdout for actual output
sys.stdout.close()
sys.stdout = sys.__stdout__

data = ast.literal_eval('''$YEARS_DICT''')
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

for crop in data.keys():
    for country, years in data[crop].items():
        if len(years) >= MIN_YEARS and country not in EXCLUDED:
            print(f'{crop}|{country}')
")

echo "Configuration:"
echo "  MIN_YEARS: $MIN_YEARS"
echo "  EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "  Maize countries: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"

# ── helpers ───────────────────────────────────────────────────────────────────

get_patchtst_flags() {
    local crop=$1
    if [ "$crop" = "wheat" ]; then
        echo "--use_gdd --use_rue --use_farquhar --use_sota_features"
    elif [ "$crop" = "maize" ]; then
        echo "--use_sota_features"
    fi
}

get_xlinear_flags() {
    echo "--use_gdd --use_rue --use_farquhar --use_sota_features --include_spatial_features --use_heat_stress_days --lag_years 2 --use_revin"
}

get_best_hps() {
    local results_file=$1
    local -n out_array=$2
    local raw_output
    raw_output=$(python3 -c "
import sys
import os

# Suppress module-level debug output
sys.stdout = open(os.devnull, 'w')
sys.path.insert(0, '../../process')
from helpers import load_best_hps

# Restore stdout for actual output
sys.stdout.close()
sys.stdout = sys.__stdout__

hps = load_best_hps('$results_file')
if not hps:
    sys.exit(1)
for k, v in hps.items():
    print(f'--{k}')
    print(f'{v}')
") || return 1
    readarray -t out_array <<< "$raw_output"
}

get_override_hps() {
    local model=$1
    local crop=$2
    local country=$3
    local -n out_array=$4

    if [ "$country" = "US" ] || [ "$country" = "BR" ]; then
        if [ "$model" = "patchtst" ] && [ "$crop" = "maize" ] && [ "$country" = "US" ]; then
            out_array=(
                "--patchtst_d_model" "128"
                "--patchtst_num_attention_heads" "8"
                "--patchtst_ffn_dim" "1024"
                "--patchtst_num_layers" "8"
                "--patchtst_dropout" "0.226374207340905"
                "--batch_size" "64"
                "--lr" "0.00021286999594907522"
                "--weight_decay" "2.0484529270261538e-05"
                "--seed" "1111"
            )
        elif [ "$model" = "patchtst" ] && [ "$crop" = "wheat" ] && [ "$country" = "US" ]; then
            out_array=(
                "--patchtst_d_model" "64"
                "--patchtst_num_attention_heads" "8"
                "--patchtst_ffn_dim" "256"
                "--patchtst_num_layers" "6"
                "--patchtst_dropout" "0.07551339306583449"
                "--batch_size" "32"
                "--lr" "7.861321845333725e-05"
                "--weight_decay" "1.5852570508768146e-06"
                "--seed" "5555"
            )
        elif [ "$model" = "patchtst" ] && [ "$crop" = "maize" ] && [ "$country" = "BR" ]; then
            out_array=(
                "--patchtst_d_model" "64"
                "--patchtst_num_attention_heads" "4"
                "--patchtst_ffn_dim" "256"
                "--patchtst_num_layers" "3"
                "--patchtst_dropout" "0.1"
                "--batch_size" "16"
                "--lr" "0.0001"
                "--weight_decay" "1e-05"
                "--seed" "42"
            )
        elif [ "$model" = "patchtst" ] && [ "$crop" = "wheat" ] && [ "$country" = "BR" ]; then
            out_array=(
                "--patchtst_d_model" "32"
                "--patchtst_num_attention_heads" "16"
                "--patchtst_ffn_dim" "128"
                "--patchtst_num_layers" "6"
                "--patchtst_dropout" "0.29463719227889157"
                "--batch_size" "16"
                "--lr" "0.0004929686671539036"
                "--weight_decay" "1.3850847658133282e-05"
                "--seed" "1111"
            )
        elif [ "$model" = "xlinear" ] && [ "$crop" = "maize" ] && [ "$country" = "US" ]; then
            out_array=(
                "--xlinear_hidden_size" "128"
                "--xlinear_temporal_ff" "128"
                "--xlinear_channel_ff" "64"
                "--xlinear_dropout" "0.27858287884282074"
                "--batch_size" "64"
                "--lr" "0.000458730339936353"
                "--weight_decay" "1.2783736085400551e-05"
                "--seed" "100"
            )
        elif [ "$model" = "xlinear" ] && [ "$crop" = "wheat" ] && [ "$country" = "US" ]; then
            out_array=(
                "--xlinear_hidden_size" "64"
                "--xlinear_temporal_ff" "64"
                "--xlinear_channel_ff" "8"
                "--xlinear_dropout" "0.44554331375563255"
                "--batch_size" "32"
                "--lr" "6.103925184472191e-05"
                "--weight_decay" "5.8262862936685296e-05"
                "--seed" "1111"
            )
        elif [ "$model" = "xlinear" ] && [ "$crop" = "maize" ] && [ "$country" = "BR" ]; then
            out_array=(
                "--xlinear_hidden_size" "32"
                "--xlinear_temporal_ff" "128"
                "--xlinear_channel_ff" "64"
                "--xlinear_dropout" "0.40907877520203867"
                "--batch_size" "16"
                "--lr" "0.00047051493707484144"
                "--weight_decay" "1.066411467206695e-06"
                "--seed" "5555"
            )
        elif [ "$model" = "xlinear" ] && [ "$crop" = "wheat" ] && [ "$country" = "BR" ]; then
            out_array=(
                "--xlinear_hidden_size" "128"
                "--xlinear_temporal_ff" "64"
                "--xlinear_channel_ff" "64"
                "--xlinear_dropout" "0.22425153841247014"
                "--batch_size" "128"
                "--lr" "0.00036225288317906414"
                "--weight_decay" "7.93191353304851e-06"
                "--seed" "1111"
            )
        else
            return 1
        fi
        return 0
    else
        return 1
    fi
}

get_countries_for_crop() {
    local crop=$1
    if   [ "$crop" = "maize" ]; then echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() { echo "$1" | tr ' ' '\n' | sort | tr '\n' ' '; }

# ── parallelism semaphore ─────────────────────────────────────────────────────

MAX_PARALLEL=2
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"
for i in $(seq 1 $MAX_PARALLEL); do echo >&3; done

run_model() {
    local log_file=$1
    shift
    local cmd=("$@")
    read -u 3
    {
        "${cmd[@]}" > "$log_file" 2>&1
        echo >&3
    } &
}

# ── final training ────────────────────────────────────────────────────────────

# Use same directory for checks and saves
CHECKPOINT_DIR="modelCheckpoints/final"
mkdir -p "$CHECKPOINT_DIR" log/final

crops=("wheat" "maize")

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop "$crop")")
    for country in $countries; do
        for model in patchtst xlinear; do

            # Check if checkpoint with this run_id already exists
            checkpoint_save_dir="${CHECKPOINT_DIR}/yield-${model}/${country}/${crop}/"
            existing_checkpoint=$(find "$checkpoint_save_dir" -name "*runid:${RUN_ID}.ckpt" 2>/dev/null | head -n 1)
            if [ -n "$existing_checkpoint" ]; then
                echo "Skipping ${model} ${country} ${crop} (run_id ${RUN_ID} already done)"
                continue
            fi

            hps_args=()
            # Try override for US/BR first, then fall back to HPO results
            if get_override_hps "$model" "$crop" "$country" hps_args; then
                echo "Using override HPS for ${model} ${country} ${crop}"
            else
                results_dir="../modelCheckpoints/results/hpo_${model}_${country}_${crop}/HPO"
                results_file=$(ls -t "${results_dir}"/*_HPO_results_*.txt 2>/dev/null | head -n 1)

                if [ -z "$results_file" ] || [ ! -s "$results_file" ]; then
                    echo "WARNING: No valid HPO results for ${model} ${country} ${crop} — skipping"
                    continue
                fi

                if ! get_best_hps "$results_file" hps_args || [ ${#hps_args[@]} -eq 0 ]; then
                    echo "ERROR: Could not load HPS for ${model} ${country} ${crop} — skipping"
                    continue
                fi
            fi

            echo "Starting final training: ${model} ${country} ${crop} (run_id: ${RUN_ID})"

            if [ "$model" = "patchtst" ]; then
                script="tstBaselines.py"
                read -ra model_flags <<< "$(get_patchtst_flags "$crop")"
            else
                script="linearBaselines.py"
                read -ra model_flags <<< "$(get_xlinear_flags)"
            fi

            cmd=(
                python "$script"
                --crop "$crop"
                --country "$country"
                --model_type "$model"
                --aggregation daily
                --epochs 100
                --use_cwb_feature
                --test_years 5
                "${model_flags[@]}"
                "${hps_args[@]}"
                --save_checkpoint_dir "$checkpoint_save_dir"
                --wandb_project "AAAI2027-CYP-HPO"
                --run_id "$RUN_ID"
            )

            run_model \
                "log/final_${RUN_ID}_${model}_${country}_${crop}.txt" \
                "${cmd[@]}"

        done
    done
done

wait
echo "Final training completed: $(date)"


