import os
import subprocess
import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QDialog, QFormLayout, QLineEdit, QLabel, \
    QDialogButtonBox, QComboBox, QFileDialog, QCheckBox
from filelock import FileLock


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("设置")

        self.lock = FileLock("training_data.json.lock")

        # 设置窗口的固定大小
        self.setFixedSize(800, 600)

        self.args_config_path = self.getRootPath() + '\\panelConfig\\argsConfig.json'
        self.load_json_data()  # 加载JSON数据并初始化界面

    # 获得根路径
    def getRootPath(self):
        # 获取文件目录
        curPath = os.path.abspath(os.path.dirname(__file__))
        # 获取项目根路径，内容为当前项目的名字
        rootPath = curPath[:curPath.find('wzry_ai') + len('wzry_ai')]
        return rootPath

    def load_json_data(self):
        with self.lock:
            """加载JSON数据"""
            with open(self.args_config_path, 'r', encoding='utf-8') as file:
                self.json_data = json.load(file)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.form_layouts = {}

        # 动态生成表单
        for section_name, settings in self.json_data.items():
            # 添加大标题
            section_label = QLabel(f"<b>{section_name}</b>")
            layout.addWidget(section_label)

            # 创建子布局，用于子内容，并设置左边距缩进
            section_layout = QVBoxLayout()
            section_layout.setContentsMargins(20, 0, 0, 0)  # 左边距设置为20px，其他边距为0
            form_layout = QFormLayout()
            section_layout.addLayout(form_layout)
            self.form_layouts[section_name] = {}

            for setting_name, setting_info in settings.items():
                help_text = setting_info["help"]
                value = setting_info["value"]
                info_text = setting_info.get("info", "")

                # 根据不同的key处理不同的逻辑
                if setting_name == "iphone_id":
                    combo_box = QComboBox()
                    device_ids = self.get_android_device_ids()
                    combo_box.addItems(device_ids)
                    help_label = QLabel(help_text)
                    if info_text:
                        help_label.setToolTip(info_text)
                    form_layout.addRow(help_label, combo_box)
                    self.form_layouts[section_name][setting_name] = combo_box

                elif setting_name == "model_path":
                    line_edit = QLineEdit(str(value))
                    select_button = QPushButton("选择路径")
                    select_button.clicked.connect(lambda: self.select_path(line_edit))
                    help_label = QLabel(help_text)
                    if info_text:
                        help_label.setToolTip(info_text)
                    form_layout.addRow(help_label, line_edit)
                    form_layout.addRow(select_button)
                    self.form_layouts[section_name][setting_name] = line_edit

                elif section_name == "是否使用gpu":
                    checkbox = QCheckBox(help_text)
                    checkbox.setChecked(value == "cuda:0")
                    if info_text:
                        checkbox.setToolTip(info_text)
                    form_layout.addRow(checkbox)
                    self.form_layouts[section_name][setting_name] = checkbox

                else:
                    line_edit = QLineEdit(str(value))
                    help_label = QLabel(help_text)
                    if info_text:
                        help_label.setToolTip(info_text)
                    form_layout.addRow(help_label, line_edit)
                    self.form_layouts[section_name][setting_name] = line_edit

            # 将子内容的布局添加到主布局
            layout.addLayout(section_layout)

        # 添加大标题
        section_label = QLabel(f"<b>设置操作</b>")
        layout.addWidget(section_label)
        # 添加刷新按钮
        refresh_button = QPushButton("刷新设置")
        refresh_button.clicked.connect(self.refresh_values)
        layout.addWidget(refresh_button)

        # 添加标准的OK和Cancel按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # 设置新的布局
        self.setLayout(layout)

    def refresh_values(self):
        """只刷新控件中的值，而不重建整个UI"""
        with self.lock:
            with open(self.args_config_path, 'r', encoding='utf-8') as file:
                new_data = json.load(file)

        for section_name, settings in new_data.items():
            for setting_name, setting_info in settings.items():
                widget = self.form_layouts[section_name][setting_name]
                new_value = setting_info["value"]

                if isinstance(widget, QLineEdit):
                    widget.setText(str(new_value))
                elif isinstance(widget, QComboBox):
                    if setting_name == "iphone_id":
                        # 重新获取设备列表并更新ComboBox
                        widget.clear()
                        device_ids = self.get_android_device_ids()
                        widget.addItems(device_ids)
                    widget.setCurrentText(str(new_value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(new_value == "cuda:0")

    def get_android_device_ids(self):
        """获取连接的Android设备的device_id"""
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
            device_lines = result.stdout.strip().split('\n')[1:]  # 第一行为标题
            device_ids = [line.split('\t')[0] for line in device_lines if '\tdevice' in line]
            return device_ids if device_ids else ["No devices connected"]
        except Exception as e:
            return ["Error fetching devices"]

    def select_path(self, line_edit):
        """打开文件对话框选择路径"""
        options = QFileDialog.Options()
        file_path = QFileDialog.getExistingDirectory(self, "选择路径", options=options)
        if file_path:
            line_edit.setText(file_path)

    def save_settings(self):
        # 遍历所有表单并更新json_data
        for section_name, settings in self.json_data.items():
            for setting_name, setting_info in settings.items():
                widget = self.form_layouts[section_name][setting_name]

                # 根据不同的key处理不同的逻辑
                if setting_name == "iphone_id":
                    setting_info["value"] = widget.currentText()

                elif setting_name == "model_path":
                    setting_info["value"] = widget.text()

                elif section_name == "是否使用gpu":
                    setting_info["value"] = "cuda:0" if widget.isChecked() else "cpu"

                else:
                    new_value = widget.text()
                    if setting_info["type"] == "int":
                        setting_info["value"] = int(new_value)
                    elif setting_info["type"] == "float":
                        setting_info["value"] = float(new_value)
                    else:
                        setting_info["value"] = new_value

        # 保存回json文件
        with self.lock:
            with open(self.args_config_path, 'w', encoding='utf-8') as file:
                json.dump(self.json_data, file, indent=4, ensure_ascii=False)
