"""
R3000 量化 MVP 系统 - 主程序入口
深色主题、K线动画播放、上帝视角标注
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6 import QtWidgets, QtGui
from config import UI_CONFIG


def main():
    """主函数"""
    # 创建应用
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建深色调色板
    palette = QtGui.QPalette()
    
    # 背景色
    palette.setColor(QtGui.QPalette.ColorRole.Window, 
                     QtGui.QColor(UI_CONFIG['THEME_BACKGROUND']))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, 
                     QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    palette.setColor(QtGui.QPalette.ColorRole.Base, 
                     QtGui.QColor(UI_CONFIG['THEME_SURFACE']))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, 
                     QtGui.QColor(UI_CONFIG['THEME_BACKGROUND']))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, 
                     QtGui.QColor(UI_CONFIG['THEME_SURFACE']))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, 
                     QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    
    # 文本色
    palette.setColor(QtGui.QPalette.ColorRole.Text, 
                     QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, 
                     QtGui.QColor('#888888'))
    
    # 按钮色
    palette.setColor(QtGui.QPalette.ColorRole.Button, 
                     QtGui.QColor(UI_CONFIG['THEME_SURFACE']))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, 
                     QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    
    # 高亮色
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, 
                     QtGui.QColor(UI_CONFIG['THEME_ACCENT']))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, 
                     QtGui.QColor('#ffffff'))
    
    # 边框/分隔线
    palette.setColor(QtGui.QPalette.ColorRole.Light, 
                     QtGui.QColor('#555555'))
    palette.setColor(QtGui.QPalette.ColorRole.Midlight, 
                     QtGui.QColor('#444444'))
    palette.setColor(QtGui.QPalette.ColorRole.Dark, 
                     QtGui.QColor('#333333'))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, 
                     QtGui.QColor('#3a3a3a'))
    palette.setColor(QtGui.QPalette.ColorRole.Shadow, 
                     QtGui.QColor('#000000'))
    
    # 链接色
    palette.setColor(QtGui.QPalette.ColorRole.Link, 
                     QtGui.QColor(UI_CONFIG['THEME_ACCENT']))
    palette.setColor(QtGui.QPalette.ColorRole.LinkVisited, 
                     QtGui.QColor('#9b59b6'))
    
    app.setPalette(palette)
    
    # 全局样式表
    app.setStyleSheet("""
        QToolTip {
            color: #cccccc;
            background-color: #2d2d2d;
            border: 1px solid #555555;
            padding: 5px;
        }
        QScrollBar:vertical {
            background: #2d2d2d;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #555555;
            min-height: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #666666;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #2d2d2d;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background: #555555;
            min-width: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #666666;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
    
    # 导入并创建主窗口
    from ui.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
