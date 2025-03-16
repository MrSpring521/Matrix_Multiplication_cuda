# 编译器设置
CC = nvcc
TARGET = main

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): main.cu
	$(CC) $< -o $@

# 清理规则
.PHONY: clean
clean:
	rm -f $(TARGET)