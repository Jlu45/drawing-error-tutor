# -*- coding: utf-8 -*-
"""
V2版本Web应用 - 智绘纠错优化版
============================
使用OrchestratorV2协调器，支持6阶段流水线和四大核心模块。
端口：5001（与V1的5000共存，方便对比测试）
"""

import os
import sys
import json
import time
import logging
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _json_serialize(data):
    return json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)

def _json_dump(data, fp, **kwargs):
    return json.dump(data, fp, ensure_ascii=False, cls=NumpyEncoder, **kwargs)

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

# 将V2目录添加到路径，确保能导入V2模块
V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '智绘纠错V2')
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from config import (
    MULTIMODAL_API_URL, MULTIMODAL_API_KEY, LLM_MODEL, VLM_MODEL,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS,
    TEXT_KNOWLEDGE_DIR, IMAGE_KNOWLEDGE_DIR, GB_STANDARDS_DIR,
    RL_EXPERIENCE_DIR, EXPERIENCE_STORE_DIR,
    FLASK_HOST, FLASK_DEBUG, VLM_JUDGE_ENABLED,
    ENABLE_ATLAS_PACK, ATLAS_CASES_PATH, ATLAS_RULES_PATH,
    ATLAS_SHOW_REFERENCE_IN_UI,
)

from orchestrator_v2 import OrchestratorV2
from rag_knowledge_base import DualKnowledgeBase

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.json_encoder = NumpyEncoder  # 全局JSON编码器，处理numpy类型
app.config['UPLOAD_FOLDER'] = 'uploads_v2'  # 使用独立的上传目录


def _sanitize_for_json(obj):
    """递归清理数据中的numpy类型，确保Jinja2 tojson和jsonify都能正常工作"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MULTIMODAL_API_URL'] = MULTIMODAL_API_URL
app.config['MULTIMODAL_API_KEY'] = MULTIMODAL_API_KEY
app.config['LLM_MODEL'] = LLM_MODEL
app.config['VLM_MODEL'] = VLM_MODEL

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(EXPERIENCE_STORE_DIR, exist_ok=True)

# 初始化知识库
kb = DualKnowledgeBase()

# 初始化V2协调器
orchestrator_v2 = OrchestratorV2(
    api_url=app.config['MULTIMODAL_API_URL'],
    api_key=app.config['MULTIMODAL_API_KEY'],
    llm_model=app.config['LLM_MODEL'],
    vlm_model=app.config.get('VLM_MODEL', ''),
    experience_dir=EXPERIENCE_STORE_DIR
)

logger = logging.getLogger('AppV2')
logging.basicConfig(level=logging.INFO)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    standard_path = os.path.join('data', 'standard_drawings', filename)
    if os.path.exists(standard_path):
        return send_from_directory(os.path.join('data', 'standard_drawings'), filename)
    return "File not found", 404


def get_recommendations(error_categories):
    suggestions = []
    if any(k in error_categories for k in ['尺寸标注', 'DIMENSION_ERROR']):
        suggestions.append('重点复习GB/T 4458.4尺寸注法标准，掌握尺寸标注的完整性和规范性')
    if any(k in error_categories for k in ['线型', '线型错误', 'LINE_TYPE_ERROR']):
        suggestions.append('熟悉GB/T 4457.4图线标准，理解实线、虚线、点画线的应用场景')
    if any(k in error_categories for k in ['公差', '公差配合', 'TOLERANCE_ERROR']):
        suggestions.append('学习公差配合基本概念，理解IT公差等级与加工精度的关系')
    if any(k in error_categories for k in ['标题栏', 'TITLE_BLOCK_ERROR']):
        suggestions.append('掌握标题栏的标准格式，了解各字段含义和填写规范')
    if any(k in error_categories for k in ['符号', '符号标注', 'SYMBOL_ERROR']):
        suggestions.append('练习基准符号、形位公差、表面粗糙度等特殊标注方法')
    if any(k in error_categories for k in ['表面粗糙度', 'ROUGHNESS_ERROR', 'SURFACE_ERROR']):
        suggestions.append('学习GB/T 131表面粗糙度标注方法，掌握代号含义和标注位置')
    if any(k in error_categories for k in ['视图标注', 'VIEW_ERROR']):
        suggestions.append('掌握GB/T 17451视图表示法，理解基本视图、向视图、局部放大图的画法')
    if any(k in error_categories for k in ['焊接符号', 'WELD_ERROR']):
        suggestions.append('学习GB/T 324焊接符号表示法，掌握基本符号和指引线规范')
    if any(k in error_categories for k in ['图幅规范', 'SHEET_ERROR']):
        suggestions.append('复习GB/T 14689图幅标准，掌握图框格式和比例选择')
    return suggestions


def get_gb_knowledge_for_errors(errors):
    """根据错误列表获取相关的GB标准知识"""
    all_gb = kb.get_all_gb_standards()
    if not all_gb:
        return []
    keywords = set()
    for error in errors:
        if isinstance(error, dict):
            etype = error.get('type', '')
            desc = error.get('description', '')
            keywords.add(etype)
            for word in [etype, desc[:10]]:
                if len(word) > 1:
                    keywords.add(word)
    relevant = []
    seen_titles = set()
    for item in all_gb:
        title = item.get('title', '')
        content = item.get('content', '')
        matched_keywords = sum(1 for kw in keywords if kw in (title + content).lower())
        if matched_keywords > 0 and title not in seen_titles:
            seen_titles.add(title)
            relevant.append({
                'title': title,
                'content': content[:800] + ('...' if len(content) > 800 else ''),
                'source': item.get('source', 'GB/T 14665-2012'),
                'matched_keywords': matched_keywords
            })
    relevant.sort(key=lambda x: x['matched_keywords'], reverse=True)
    return relevant[:5]


@app.route('/')
def index():
    """V2首页 - 上传界面"""
    return render_template('index_v2.html',
                         version='V2',
                         port=5001,
                         features=[
                             '6阶段流水线 (Planning→Perception→RuleCheck→Analysis→Fusion→Judge)',
                             '预检机制 (Pre-check Protocol)',
                             '跨阶段定向修复 (Cross-Stage Rollback)',
                             '非对称经验存储 (Experience Store)',
                             'VLM质量评审 (4维连续评分)',
                             '纯numpy实现的RL记忆单元'
                         ])


@app.route('/upload', methods=['POST'])
def upload():
    """上传图纸并使用V2分析"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # 保存文件
    timestamp = int(time.time() * 1000)
    filename = f"v2_{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    logger.info(f"[V2] 文件已保存: {filepath}")

    # 执行V2分析（6阶段流水线）
    start_time = time.time()

    try:
        # 获取背景知识
        background_knowledge = kb.get_background_knowledge_text()

        # 是否启用VLM评审（可通过参数控制）
        enable_judge = request.form.get('enable_judge', 'false').lower() == 'true'

        # 执行V2完整分析
        result = orchestrator_v2.analyze(
            image_path=filepath,
            background_knowledge=background_knowledge,
            enable_judge=enable_judge and VLM_JUDGE_ENABLED
        )

        # 递归清理numpy类型（numpy.bool_等无法被Jinja2 tojson序列化）
        result = _sanitize_for_json(result)

        analysis_time = time.time() - start_time

        # 提取结果
        errors = result.get('errors', [])
        ocr_results = result.get('ocr_results', [])
        detection_results = result.get('detection_results', [])
        feedback = result.get('feedback', [])
        report_data = result.get('report', {})
        contract_validation = result.get('contract_validation', {})

        # 获取GB知识
        gb_knowledge = get_gb_knowledge_for_errors(errors)

        # 获取API结果
        api_result = result.get('api_result', {})
        raw_response = api_result.get('raw_response', '')
        model_name = api_result.get('model', 'unknown')
        logger.info(f"[V2] api_result: model={model_name}, raw_response长度={len(raw_response)}")
        if not raw_response:
            logger.warning(f"[V2] api_result.raw_response为空! api_result keys={list(api_result.keys()) if api_result else 'None'}")
            logger.warning(f"[V2] result keys={list(result.keys())}")

        # 获取指标
        metrics = result.get('metrics', {})
        phase_timings = metrics.get('phase_timings', {})
        # 将contract_stats的status_counts扁平化，供模板直接读取
        raw_contract_stats = metrics.get('contract_stats', {})
        status_counts = raw_contract_stats.get('status_counts', {})
        contract_stats = {
            'total': raw_contract_stats.get('total', 0),
            'completed': status_counts.get('completed', 0),
            'failed': status_counts.get('failed', 0),
            'pending': status_counts.get('pending', 0),
            'in_progress': status_counts.get('in_progress', 0),
            'avg_confidence': raw_contract_stats.get('avg_confidence', 0),
            'total_exec_time_ms': raw_contract_stats.get('total_exec_time_ms', 0)
        }
        rollback_stats = metrics.get('rollback_stats', {})
        experience_stats = metrics.get('experience_stats', {})
        rl_stats = metrics.get('rl_stats')

        # 构建V2特有信息
        # 从contract_list构建预检详情（使用真实状态）
        contract_details = []
        for c in metrics.get('contract_list', []):
            status_text = c.get('status', 'pending')
            if status_text == 'completed':
                icon = '✓'
                label = '已完成'
                css_class = 's-completed'
            elif status_text == 'failed':
                icon = '✗'
                label = '失败'
                css_class = 's-failed'
            elif status_text == 'in_progress':
                icon = '◐'
                label = '进行中'
                css_class = 's-pending'
            else:
                icon = '○'
                label = '待处理'
                css_class = 's-pending'
            contract_details.append({
                'name': f"{c.get('category', '?')} ({c.get('detector', '?')})",
                'category': c.get('category', '?'),
                'detector': c.get('detector', '?'),
                'status': status_text,
                'icon': icon,
                'label': label,
                'css_class': css_class,
                'confidence': c.get('confidence', 0),
                'execution_time_ms': c.get('execution_time_ms', 0)
            })

        v2_info = {
            'contract_validation': contract_validation,
            'contract_details': contract_details,
            'phase_timings': phase_timings,
            'contract_stats': contract_stats,
            'rollback_stats': rollback_stats,
            'experience_stats': experience_stats,
            'rl_stats': rl_stats,
            'judge_result': result.get('judge_result'),
            'api_result': result.get('api_result', {})
        }

        # 将V2的report包装成V1模板期望的格式
        # V1模板期望: report.summary.overall_score, report.summary.total_errors 等
        # V2返回: report = {total_errors, error_categories, overall_score, summary}
        # 添加llm_summary字段供模板使用
        if 'llm_summary' not in report_data:
            report_data['llm_summary'] = report_data.get('summary', '')
        report = {
            'filename': filename,
            'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': report_data,
            'error_details': [
                {
                    'id': i + 1,
                    'type': e.get('type', '未知'),
                    'description': e.get('description', ''),
                    'suggestion': e.get('suggestion', ''),
                    'severity': e.get('severity', '中'),
                    'source': e.get('source', 'rule_check'),
                    'gb_reference': e.get('gb_reference', '')
                }
                for i, e in enumerate(errors)
            ],
            'recommendations': get_recommendations(report_data.get('error_categories', {})),
            'gb_knowledge': gb_knowledge
        }

        # 保存会话信息到session或文件
        session_id = f"v2_{timestamp}"
        session_data = {
            'session_id': session_id,
            'filepath': filepath,
            'result': result,
            'analysis_time': analysis_time
        }
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            _json_dump({
                'errors': errors,
                'report': report,
                'v2_info': v2_info,
                'model': model_name
            }, f, indent=2)

        logger.info(f"[V2] 分析完成: {len(errors)}个错误, 耗时{analysis_time:.2f}秒")

        return render_template('result_v2.html',
                             version='V2',
                             port=5001,
                             filename=filename,
                             filepath=filepath,
                             errors=errors,
                             ocr_results=ocr_results,
                             detection_results=detection_results,
                             feedback=feedback,
                             report=report,
                             gb_knowledge=gb_knowledge,
                             api_result=api_result,
                             raw_response=raw_response[:3000] if raw_response else '',
                             model=model_name,
                             analysis_time=round(analysis_time, 2),
                             session_id=session_id,
                             rl_session_id=session_id,
                             v2_info=v2_info)

    except Exception as e:
        logger.error(f"[V2] 分析失败: {e}", exc_info=True)
        import traceback
        error_detail = traceback.format_exc()

        return render_template('result_v2.html',
                             version='V2',
                             port=5001,
                             filename=filename,
                             filepath=filepath,
                             errors=[{
                                 'type': '系统错误',
                                 'description': f'分析过程中发生错误: {str(e)}',
                                 'severity': '高',
                                 'suggestion': '请检查日志获取详细信息'
                             }],
                             ocr_results=[],
                             detection_results=[],
                             feedback=['系统遇到错误，请重试'],
                             report={
                                 'filename': filename,
                                 'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'summary': {
                                     'total_errors': 1,
                                     'error_categories': {'系统错误': 1},
                                     'overall_score': 0,
                                     'summary': '分析失败',
                                     'llm_summary': '分析失败'
                                 },
                                 'error_details': [{
                                     'id': 1,
                                     'type': '系统错误',
                                     'description': f'分析过程中发生错误: {str(e)}',
                                     'suggestion': '请检查日志获取详细信息',
                                     'severity': '高',
                                     'source': 'system',
                                     'gb_reference': ''
                                 }],
                                 'recommendations': [],
                                 'gb_knowledge': []
                             },
                             gb_knowledge=[],
                             api_result={'raw_response': error_detail[:2000], 'model': 'error'},
                             raw_response=error_detail[:2000],
                             model='error',
                             analysis_time=0,
                             session_id='error',
                             rl_session_id='error',
                             v2_info={})


@app.route('/api/rl_feedback', methods=['POST'])
def rl_feedback():
    """V2版本的RL反馈接口（同时驱动RL和Experience Store）"""
    data = request.json or {}
    session_id = data.get('session_id', '')
    error_index = data.get('error_index', 0)
    action = data.get('action', 'ignored')  # confirmed / ignored / dismissed_all / partial_confirm / useful_guidance

    # 获取额外元数据（用于Experience Store）
    drawing_type = data.get('drawing_type', '')
    error_category = data.get('error_category', '')
    detection_result = _json_serialize(data.get('detection_result', {}))
    correction = data.get('correction', '')
    accuracy = float(data.get('accuracy', 0.0))
    helpfulness = float(data.get('helpfulness', 0.0))

    # 注册反馈到OrchestratorV2（同时驱动RL和Experience Store）
    orchestrator_v2.register_feedback(
        session_id=session_id,
        feedback_type=action,
        drawing_type=drawing_type,
        error_category=error_category,
        detection_result=detection_result,
        correction=correction,
        accuracy=accuracy,
        helpfulness=helpfulness
    )

    # 计算奖励
    reward_map = {
        'confirmed': 1.0,
        'useful_guidance': 0.5,
        'partial_confirm': 0.3,
        'ignored': -0.5,
        'dismissed_all': -1.0
    }
    reward = reward_map.get(action, 0.0)

    logger.info(f"[V2] RL反馈: session={session_id}, action={action}, reward={reward}")

    return jsonify({
        'status': 'success',
        'reward': reward,
        'action': action,
        'message': f'V2反馈已接收: {action} (同时更新RL和Experience Store)'
    })


@app.route('/api/rl_stats')
def rl_stats():
    """统计接口（包含RL + Experience Store + 预检统计）"""
    rl_stats = None
    if orchestrator_v2._rl_memory:
        rl_stats = orchestrator_v2._rl_memory.get_stats()

    return jsonify({
        'version': 'V2',
        'rl_memory': rl_stats,
        'experience_store': orchestrator_v2.experience.stats,
        'orchestrator': orchestrator_v2.stats,
        'total_analyses': orchestrator_v2.stats.get('total_analyses', 0),
        'total_rollbacks': orchestrator_v2.stats.get('total_rollbacks', 0)
    })


@app.route('/api/experience_stats')
def experience_stats():
    """Experience Store统计接口"""
    return jsonify({
        'version': 'V2',
        'experience_store': orchestrator_v2.experience.stats,
        'cases_by_role': {
            role.value: len(orchestrator_v2.experience._cases_by_role.get(role, []))
            for role in ['PLANNING', 'PERCEPTION', 'ANALYSIS', 'JUDGING']
            if role in [type(r) for r in orchestrator_v2.experience._cases_by_role.keys()]
        } if hasattr(orchestrator_v2.experience, '_cases_by_role') else {}
    })


@app.route('/compare')
def compare():
    """对比页面 - V1 vs V2"""
    return render_template('compare.html',
                         v1_url='http://127.0.0.1:5000',
                         v2_url='http://127.0.0.1:5001')


# ==================== Atlas 图册能力包 API ====================

_atlas_registry_instance = None


def _get_atlas_registry():
    global _atlas_registry_instance
    if _atlas_registry_instance is not None:
        return _atlas_registry_instance
    if ENABLE_ATLAS_PACK:
        try:
            from atlas.atlas_registry import AtlasRegistry
            _atlas_registry_instance = AtlasRegistry(ATLAS_CASES_PATH, ATLAS_RULES_PATH)
            return _atlas_registry_instance
        except Exception as e:
            logger.warning(f"Atlas Registry 初始化失败: {e}")
    return None


@app.route('/api/atlas/case/<case_id>')
def get_atlas_case(case_id):
    """获取图册案例详情"""
    registry = _get_atlas_registry()
    if not registry:
        return jsonify({'error': 'Atlas 图册功能未启用'}), 404

    case = registry.get_case(case_id)
    if not case:
        return jsonify({'error': f'案例 {case_id} 不存在'}), 404

    base_dir = os.path.dirname(os.path.abspath(__file__))
    v2_dir = os.path.join(base_dir, '智绘纠错V2')

    wrong_img = case.get('wrong_image', '')
    correct_img = case.get('correct_image', '')
    case_img = os.path.join('data', 'atlas', 'images', 'cases', f'{case_id}.png')

    if wrong_img and not os.path.isabs(wrong_img):
        wrong_img = os.path.join(v2_dir, wrong_img)
    if correct_img and not os.path.isabs(correct_img):
        correct_img = os.path.join(v2_dir, correct_img)
    if not os.path.isabs(case_img):
        case_img = os.path.join(v2_dir, case_img)

    import base64
    wrong_b64 = ''
    correct_b64 = ''
    case_b64 = ''

    try:
        if wrong_img and os.path.exists(wrong_img):
            with open(wrong_img, 'rb') as f:
                wrong_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        pass

    try:
        if correct_img and os.path.exists(correct_img):
            with open(correct_img, 'rb') as f:
                correct_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        pass

    try:
        if case_img and os.path.exists(case_img):
            with open(case_img, 'rb') as f:
                case_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        pass

    return jsonify({
        'case_id': case.get('case_id', ''),
        'chapter': case.get('chapter', ''),
        'section': case.get('section', ''),
        'figure_no': case.get('figure_no', ''),
        'case_name': case.get('case_name', ''),
        'v2_error_category': case.get('v2_error_category', ''),
        'source_text': case.get('source_text', ''),
        'keywords': case.get('keywords', []),
        'teaching_hint': case.get('teaching_hint', ''),
        'suggestion': case.get('suggestion', ''),
        'wrong_image_b64': wrong_b64,
        'correct_image_b64': correct_b64,
        'case_image_b64': case_b64,
    })


@app.route('/api/atlas/rules')
def get_atlas_rules():
    """获取所有启用的图册规则"""
    registry = _get_atlas_registry()
    if not registry:
        return jsonify({'error': 'Atlas 图册功能未启用'}), 404

    rules = registry.get_active_rules()
    return jsonify({
        'total': len(rules),
        'rules': [
            {
                'rule_id': r.get('rule_id', ''),
                'name': r.get('name', ''),
                'v2_error_category': r.get('v2_error_category', ''),
                'priority': r.get('priority', ''),
                'check_type': r.get('check_type', ''),
                'enabled': r.get('enabled', True),
                'message': r.get('message', ''),
            }
            for r in rules
        ]
    })


@app.route('/api/atlas/feedback', methods=['POST'])
def atlas_feedback():
    """教师反馈接口"""
    data = request.json or {}
    case_id = data.get('case_id', '')
    rule_id = data.get('rule_id', '')
    action = data.get('action', '')
    note = data.get('note', '')

    feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '智绘纠错V2', 'data', 'atlas')
    feedback_file = os.path.join(feedback_dir, 'atlas_rule_feedback.jsonl')

    try:
        os.makedirs(feedback_dir, exist_ok=True)
        import datetime
        feedback_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'case_id': case_id,
            'rule_id': rule_id,
            'action': action,
            'note': note,
        }
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.warning(f"Atlas 反馈写入失败: {e}")

    return jsonify({'success': True, 'message': f'反馈已记录: {action}'})


@app.route('/api/atlas/stats')
def atlas_stats():
    """Atlas 统计信息"""
    registry = _get_atlas_registry()
    if not registry:
        return jsonify({'enabled': False})

    return jsonify({
        'enabled': True,
        'total_cases': len(registry.cases),
        'total_rules': len(registry.rules),
        'active_rules': len(registry.get_active_rules()),
        'categories': list(registry._category_rules.keys()),
    })


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════╗
║          智绘纠错 V2 版本 Web 服务                          ║
║  ─────────────────────────────────────────────────────────── ║
║  地址: http://127.0.0.1:5001                               ║
║  特性: 6阶段流水线 | 四大核心模块 | VLM质量评审           ║
║  对比: V1版本运行在 http://127.0.0.1:5000                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    app.run(host=FLASK_HOST, port=5001, debug=FLASK_DEBUG)
