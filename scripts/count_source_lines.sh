#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '用法: %s [目录]\n' "$(basename "$0")" >&2
}

if [[ "$#" -gt 1 ]]; then
    usage
    exit 2
fi

# 不传参数时，使用运行脚本时所在的目录。
target_dir="${1:-$(pwd)}"

if [[ ! -d "$target_dir" ]]; then
    printf '错误: 目录不存在: %s\n' "$target_dir" >&2
    exit 1
fi

# 输出：文件数、空行数、注释行数、代码行数。
count_language() {
    local comment_style="$1"
    shift
    local files stats

    files=$(find "$target_dir" -type f \( "$@" \) -print0 \
        | awk -v RS='\0' 'END { print NR + 0 }')

    stats=$(find "$target_dir" -type f \( "$@" \) -exec awk -v style="$comment_style" '
        function trim(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            return value
        }

        {
            line = $0
            had_comment = 0

            if (style == "c") {
                while (line ~ /\/\*/) {
                    had_comment = 1
                    before = line
                    sub(/\/\*.*/, "", before)
                    after = line
                    sub(/^.*\/\*/, "", after)

                    if (line ~ /\/\*([^*]|\*+[^*/])*\*+\//) {
                        closing = after
                        sub(/([^*]|\*+[^*/])*\*+\//, "", closing)
                        line = before closing
                    } else {
                        line = before
                        in_block_comment = 1
                        break
                    }
                }

                if (in_block_comment) {
                    if (line ~ /\*\//) {
                        sub(/^.*\*\//, "", line)
                        in_block_comment = 0
                    } else {
                        line = ""
                    }
                    had_comment = 1
                }

                if (!in_block_comment && line ~ /\/\//) {
                    sub(/\/\/.*/, "", line)
                    had_comment = 1
                }
            } else {
                if (line ~ /^[[:space:]]*#/) {
                    line = ""
                    had_comment = 1
                } else if (line ~ /#/) {
                    sub(/#.*/, "", line)
                    had_comment = 1
                }
            }

            line = trim(line)
            if (line == "") {
                if (had_comment) comment++
                else blank++
            } else {
                code++
            }
        }

        END { printf "%d %d %d\n", blank + 0, comment + 0, code + 0 }
    ' {} + | awk '
        { blank += $1; comment += $2; code += $3 }
        END { printf "%d %d %d", blank + 0, comment + 0, code + 0 }
    ')

    printf '%s %s\n' "$files" "$stats"
}

read -r c_files c_blank c_comment c_code < <(count_language c \
    -iname '*.c' -o -iname '*.h')
read -r cpp_files cpp_blank cpp_comment cpp_code < <(count_language c \
    -iname '*.cc' -o -iname '*.cpp' -o -iname '*.cxx' \
    -o -iname '*.hh' -o -iname '*.hpp' -o -iname '*.hxx')
read -r cuda_files cuda_blank cuda_comment cuda_code < <(count_language c \
    -iname '*.cu' -o -iname '*.cuh')
read -r python_files python_blank python_comment python_code < <(count_language python \
    -iname '*.py')

sum_files=$((c_files + cpp_files + cuda_files + python_files))
sum_blank=$((c_blank + cpp_blank + cuda_blank + python_blank))
sum_comment=$((c_comment + cpp_comment + cuda_comment + python_comment))
sum_code=$((c_code + cpp_code + cuda_code + python_code))

printf '%s\n' '-------------------------------------------------------'
printf '%-14s %8s %10s %11s %8s\n' 'Language' 'files' 'blank' 'comment' 'code'
printf '%s\n' '-------------------------------------------------------'
printf '%-14s %8d %10d %11d %8d\n' 'C' "$c_files" "$c_blank" "$c_comment" "$c_code"
printf '%-14s %8d %10d %11d %8d\n' 'C++' "$cpp_files" "$cpp_blank" "$cpp_comment" "$cpp_code"
printf '%-14s %8d %10d %11d %8d\n' 'CUDA' "$cuda_files" "$cuda_blank" "$cuda_comment" "$cuda_code"
printf '%-14s %8d %10d %11d %8d\n' 'Python' "$python_files" "$python_blank" "$python_comment" "$python_code"
printf '%s\n' '-------------------------------------------------------'
printf '%-14s %8d %10d %11d %8d\n' 'Sum' "$sum_files" "$sum_blank" "$sum_comment" "$sum_code"
printf '%s\n' '-------------------------------------------------------'
