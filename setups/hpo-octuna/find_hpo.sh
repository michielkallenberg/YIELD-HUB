#!/bin/bash
#SBATCH --job-name=cybench_hpo_new
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100
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

MIN_YEARS=8
EXCLUDED_COUNTRIES=""

YEARS_DICT=$(cat <<'EOF'
{
  "maize": {
    "AO": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "AR": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "AT": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "BE": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "BF": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019],
    "BG": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "CN": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "CZ": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "DE": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "DK": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "EE": [],
    "EL": [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "ES": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "ET": [2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2020],
    "FI": [],
    "FR": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "HR": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "HU": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "IE": [],
    "IN": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "IT": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "LS": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "LT": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "LV": [],
    "MG": [2005, 2006, 2007, 2008, 2009, 2010],
    "ML": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "MW": [2018, 2019, 2020, 2021, 2022, 2023],
    "MX": [2014, 2017, 2019, 2022],
    "MZ": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "NE": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2020, 2021],
    "NL": [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "PL": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "PT": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "RO": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "SE": [2007, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    "SK": [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    "SN": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    "TD": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "ZA": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "ZM": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
  },
  "wheat": {
    "AR": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "AT": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "AU": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "BE": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "BG": [2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    "CN": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "CZ": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "DE": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "DK": [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "EE": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "EL": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "ES": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "FI": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "FR": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "HR": [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "HU": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "IE": [2010, 2011, 2012, 2019, 2020],
    "IN": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "IT": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "LT": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "LV": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    "NL": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "PL": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "PT": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "RO": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "SE": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "SK": [2017, 2018]
  }
}
EOF
)

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if [ "$crop" = "maize" ]; then
        MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then
        WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import ast

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
echo "  Maize countries with >=$MIN_YEARS years: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries with >=$MIN_YEARS years: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"


MAX_PARALLEL=6

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"

for i in $(seq 1 $MAX_PARALLEL); do
    echo >&3
done

mkdir -p modelCheckpoints/in-season/results modelCheckpoints/in-season/hpo log

# Acquires a token, launches process, releases token when done
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


is_completed() {
    local model=$1
    local country=$2
    local crop=$3
    local results_dir="modelCheckpoints/results/hpo_${model}_${country}_${crop}/HPO"

    # Check if HPO directory exists and has any .txt files with "HPO COMPLETED"
    if [ -d "$results_dir" ]; then
        # Look for any .txt file in the HPO directory that contains "HPO COMPLETED"
        for file in "$results_dir"/*.txt; do
            if [ -f "$file" ] && grep -q "HPO COMPLETED" "$file" 2>/dev/null; then
                return 0  # Completed - HPO finished
            fi
        done
    fi

    return 1  # Not completed - no HPO COMPLETED found
}

get_countries_for_crop() {
    local crop=$1
    if [ "$crop" = "maize" ]; then
        echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then
        echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() {
    echo $1 | tr ' ' '\n' | sort | tr '\n' ' '
}

crops=("wheat" "maize")

# HPO: patchtst
# echo "--------------------------------------"
# echo "Running HPO for patchtst (HF)"
# echo "--------------------------------------"

# for crop in "${crops[@]}"; do
#     countries=$(sort_countries "$(get_countries_for_crop $crop)")
#     for country in $countries; do

#         if is_completed "patchtst" "$country" "$crop"; then
#             echo "Skipping patchtst $country $crop (already completed)"
#             continue
#         fi

#         results_dir="modelCheckpoints/results/hpo_patchtst_${country}_${crop}"
#         echo "Starting HPO: patchtst $country $crop"

#         # Crop-specific feature flags
#         patchtst_flags=()
#         if [ "$crop" = "wheat" ]; then
#             patchtst_flags=(--use_gdd --use_rue --use_farquhar --use_sota_features)
#         elif [ "$crop" = "maize" ]; then
#             patchtst_flags=(--use_sota_features)
#         fi

#         cmd=(
#             python tstBaselines.py
#             --crop "$crop"
#             --country "$country"
#             --model_type patchtst
#             --aggregation daily
#             --epochs 15
#             --lag_years 0
#             --use_cwb_feature
#             --test_years 5
#             "${patchtst_flags[@]}"
#             --n_trials 50
#             --hpo_objective nrmse
#             --hpo_study_name "AAAI2027-HPO"
#             --save_checkpoint_dir "modelCheckpoints/in-season/yield-patchtst-hpo/$country/$crop/"
#             --forecast_type "end-of-season"
#             --results_dir "$results_dir"
#         )

#         run_model \
#             "modelCheckpoints/in-season/hpo/hpo_patchtst_${country}_${crop}.txt" \
#             "${cmd[@]}"

#     done
# done

# HPO: Xlinear
echo "--------------------------------------"
echo "Running HPO for xlinear (Linear)"
echo "--------------------------------------"

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop $crop)")
    for country in $countries; do

        if is_completed "xlinear" "$country" "$crop"; then
            echo "Skipping xlinear $country $crop (already completed)"
            continue
        fi

        results_dir="modelCheckpoints/results/hpo_xlinear_${country}_${crop}"
        echo "Starting HPO: xlinear $country $crop"

        # Crop-specific feature flags (same for wheat and maize)
        xlinear_flags=(
            --use_gdd --use_rue --use_farquhar --use_sota_features
            --include_spatial_features --use_heat_stress_days
            --lag_years 2 --use_revin
        )

        cmd=(
            python linearBaselines.py
            --crop "$crop"
            --country "$country"
            --model_type xlinear
            --aggregation daily
            --epochs 15
            --use_cwb_feature
            --test_years 5
            "${xlinear_flags[@]}"
            --n_trials 50
            --hpo_objective nrmse
            --hpo_study_name "AAAI2027-HPO"
            --save_checkpoint_dir "modelCheckpoints/in-season/yield-xlinear-hpo/$country/$crop/"
            --results_dir "$results_dir"
            --forecast_type "middle-of-season"
        )

        run_model \
            "modelCheckpoints/in-season/hpo/hpo_xlinear_${country}_${crop}.txt" \
            "${cmd[@]}"

    done
done

wait

echo "End time: $(date)"
echo "All HPO jobs finished."