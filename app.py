import os
import json
from dotenv import load_dotenv
import sys
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from datetime import datetime
import logging

# 加载环境变量
load_dotenv()
WORKDIR = os.getenv("WORKDIR")
if WORKDIR:
    os.chdir(WORKDIR)
    sys.path.append(WORKDIR)

from langchain_core.messages import HumanMessage
# 确保你的 LangGraph 代码在 src/agent.py 中，并且有名为 app 的编译后的图
from src.agent import app as agent_app

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
flask_app = Flask(__name__, template_folder='templates')
flask_app.secret_key = 'sepsis-cdss-secret-key' # 修改为相关的密钥
CORS(flask_app)

# 存储会话数据 (生产环境建议使用 Redis 或数据库)
sessions_store = {}

def get_thread_id(session_id):
    """获取或创建线程ID"""
    if session_id not in sessions_store:
        sessions_store[session_id] = {
            'thread_id': f"thread_{session_id}",
            'messages': []
        }
    return sessions_store[session_id]['thread_id']

def get_session_messages(session_id):
    """获取会话消息"""
    if session_id not in sessions_store:
        sessions_store[session_id] = {
            'thread_id': f"thread_{session_id}",
            'messages': []
        }
    return sessions_store[session_id]['messages']

@flask_app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@flask_app.route('/api/chat', methods=['POST'])
def chat():
    """聊天API端点"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # 获取线程ID和消息历史
        thread_id = get_thread_id(session_id)
        messages = get_session_messages(session_id)
        
        # 添加用户消息到历史
        user_msg = {
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(user_msg)
        
        # 通过 Agent 处理消息
        response_messages = []
        try:
            # 调用 LangGraph Agent
            for event in agent_app.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config={"configurable": {"thread_id": thread_id}}
            ):
                # 提取 Agent 的自然语言响应
                if "agent" in event:
                    agent_response = event["agent"]
                    if "messages" in agent_response:
                        # 假设最后一条消息是给用户的回复
                        msg = agent_response["messages"][-1]
                        if hasattr(msg, 'content') and msg.content:
                            # 覆盖式更新，只保留最新的回复作为最终输出
                            # 如果你想保留流式过程中的每一句话，可以用 append
                             response_messages.append(msg.content)
            
            # 获取最后的 AI 响应
            ai_response = response_messages[-1] if response_messages else "Analysis complete provided via tools."
            
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}")
            ai_response = f"System Error: {str(e)}. Please check patient data connection."
        
        # 添加 AI 响应到历史
        ai_msg = {
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(ai_msg)
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id,
            'thread_id': thread_id
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/history', methods=['GET'])
def get_history():
    """获取聊天历史"""
    try:
        session_id = request.args.get('session_id', 'default')
        messages = get_session_messages(session_id)
        return jsonify({
            'messages': messages,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/clear', methods=['POST'])
def clear_history():
    """清除聊天历史"""
    try:
        session_id = request.json.get('session_id', 'default')
        if session_id in sessions_store:
            # 重置时生成新的 thread_id 以清除 LangGraph 记忆
            sessions_store[session_id] = {
                'thread_id': f"thread_{session_id}_{int(datetime.now().timestamp())}",
                'messages': []
            }
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    return jsonify({
        'status': 'online',
        'system': 'Sepsis Clinical Decision Support System',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 确保 template 文件夹存在
    if not os.path.exists(os.path.join(WORKDIR, 'templates')):
         os.makedirs(os.path.join(WORKDIR, 'templates'))
         print("Created templates directory. Please place index.html inside it.")
         
    flask_app.run(debug=True, host='0.0.0.0', port=5001)


    