from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import json
import numpy as np
import time
import logging
from src.rag_knowledge_base import DualKnowledgeBase
from src.multi_agent_system import DrawingOrchestrator, OrchestratorV2
from config_loader import (
    MULTIMODAL_API_URL, MULTIMODAL_API_KEY, LLM_MODEL,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    VLM_MODEL, VLM_JUDGE_ENABLED, ENABLE_ATLAS_PACK,
    ATLAS_CASES_PATH, ATLAS_RULES_PATH, ATLAS_SHOW_REFERENCE_IN_UI,
    EXPERIENCE_STORE_DIR
)

logger = logging.getLogger("App")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

app.config['MULTIMODAL_API_URL'] = MULTIMODAL_API_URL
app.config['MULTIMODAL_API_KEY'] = MULTIMODAL_API_KEY
app.config['LLM_MODEL'] = LLM_MODEL

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

kb = DualKnowledgeBase()

orchestrator = DrawingOrchestrator(
    api_url=app.config['MULTIMODAL_API_URL'],
    api_key=app.config['MULTIMODAL_API_KEY'],
    llm_model=app.config['LLM_MODEL']
)

orchestrator_v2 = OrchestratorV2(
    api_url=app.config['MULTIMODAL_API_URL'],
    api_key=app.config['MULTIMODAL_API_KEY'],
    llm_model=app.config['LLM_MODEL'],
    vlm_api_url=app.config['MULTIMODAL_API_URL'],
    vlm_api_key=app.config['MULTIMODAL_API_KEY'],
    vlm_model=VLM_MODEL
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_gb_knowledge_for_errors(errors):
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
        score = 0
        for kw in keywords:
            if kw in title:
                score += 3
            if kw in content:
                score += 1
        if score > 0 and title not in seen_titles:
            relevant.append(item)
            seen_titles.add(title)
    if not relevant:
        relevant = all_gb[:3]
    return relevant

def get_recommendations(error_categories):
    suggestions = []
    if '尺寸标注' in error_categories:
        suggestions.append('重点复习GB/T 4458.4尺寸注法标准，掌握尺寸标注的完整性和规范性')
    if '线型' in error_categories:
        suggestions.append('熟悉GB/T 4457.4图线标准，理解实线、虚线、点画线的应用场景')
    if '公差' in error_categories:
        suggestions.append('学习公差配合基本概念，理解IT公差等级与加工精度的关系')
    if '标题栏' in error_categories:
        suggestions.append('掌握标题栏的标准格式，了解各字段含义和填写规范')
    if '符号' in error_categories:
        suggestions.append('练习基准符号、形位公差、表面粗糙度等特殊标注方法')
    if '焊接符号' in error_categories:
        suggestions.append('学习GB/T 324焊缝符号表示法，掌握焊接符号的标注规则和指引线要求')
    if '图幅规范' in error_categories:
        suggestions.append('复习GB/T 14689图纸幅面和格式标准，掌握图框、比例、图幅选用规则')
    if '表面粗糙度' in error_categories:
        suggestions.append('学习GB/T 131表面粗糙度标注方法，理解Ra参数含义和标注方向要求')
    if '视图标注' in error_categories:
        suggestions.append('掌握GB/T 4458.1图样画法标准，理解视图投影方向、局部放大图和剖视图标注')
    return suggestions

# ==================== V1 Routes ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        background_knowledge = kb.get_background_knowledge_text(2000)

        result = orchestrator.analyze(filepath, background_knowledge)

        ocr_results = result['ocr_results']
        detection_results = result['detection_results']
        errors = result['errors']
        feedback = result['feedback']
        api_result = result['api_result']
        geo_result = result.get('geo_result')
        structure_result = result.get('structure_result')
        report_data = result['report']

        gb_knowledge = get_gb_knowledge_for_errors(errors)

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

        geo_summary = None
        if geo_result:
            parts = []
            lines = geo_result.get('lines', [])
            circles = geo_result.get('circles', [])
            arrows = geo_result.get('arrows', [])
            lt = geo_result.get('line_types', {})
            dim_structs = geo_result.get('dimension_structures', [])
            contours = geo_result.get('contours', [])
            parts.append(f"直线: {len(lines)}条")
            parts.append(f"圆: {len(circles)}个")
            parts.append(f"箭头: {len(arrows)}个")
            parts.append(f"尺寸线对: {len(dim_structs)}对")
            parts.append(f"轮廓形状: {len(contours)}个")
            parts.append(f"线型: 实线{lt.get('solid_count',0)}/虚线{lt.get('dashed_count',0)}/点画线{lt.get('center_line_count',0)}")
            geo_summary = '；'.join(parts)

        rl_session_id = result.get('metrics', {}).get('rl_session_id', '')
        rl_stats = result.get('metrics', {}).get('rl_stats', {})

        return render_template('result.html',
                           filename=filename,
                           ocr_results=ocr_results,
                           detection_results=detection_results,
                           errors=errors,
                           feedback=feedback,
                           gb_knowledge=gb_knowledge,
                           api_result=api_result,
                           report=report,
                           geo_summary=geo_summary,
                           rl_session_id=rl_session_id,
                           rl_stats=rl_stats)

    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    standard_path = os.path.join('data', 'standard_drawings', filename)
    if os.path.exists(standard_path):
        return send_from_directory(os.path.join('data', 'standard_drawings'), filename)
    return "File not found", 404

@app.route('/api/gb_standards')
def api_gb_standards():
    all_gb = kb.get_all_gb_standards()
    query = request.args.get('q', '')
    if query:
        results = kb.search_gb_standards(query)
    else:
        results = all_gb
    return json.dumps(results, ensure_ascii=False, default=str)

@app.route('/api/rl_feedback', methods=['POST'])
def rl_feedback():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No JSON data'}), 400

    session_id = data.get('session_id', '')
    error_description = data.get('error_description', '')
    feedback_type = data.get('feedback_type', '')

    if feedback_type not in ('confirmed', 'ignored', 'dismissed_all', 'partial_confirm', 'useful_guidance'):
        return jsonify({'success': False, 'error': f'Invalid feedback_type: {feedback_type}'}), 400

    if not session_id:
        return jsonify({'success': False, 'error': 'Missing session_id'}), 400

    try:
        orchestrator.rl_memory.submit_feedback(session_id, error_description, feedback_type)
        stats = orchestrator.rl_memory.get_stats()
        return jsonify({
            'success': True,
            'message': f'Feedback "{feedback_type}" recorded for session {session_id}',
            'rl_stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rl_stats')
def rl_stats():
    return jsonify(orchestrator.rl_memory.get_stats())

# ==================== V2 Routes ====================

@app.route('/v2')
def index_v2():
    return render_template('index_v2.html')

@app.route('/v2/upload', methods=['POST'])
def upload_file_v2():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        background_knowledge = kb.get_background_knowledge_text(2000)

        result = orchestrator_v2.analyze(filepath, background_knowledge)

        ocr_results = result['ocr_results']
        detection_results = result['detection_results']
        errors = result['errors']
        feedback = result['feedback']
        api_result = result['api_result']
        geo_result = result.get('geo_result')
        structure_result = result.get('structure_result')
        report_data = result['report']

        gb_knowledge = get_gb_knowledge_for_errors(errors)

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

        geo_summary = None
        if geo_result:
            parts = []
            lines = geo_result.get('lines', [])
            circles = geo_result.get('circles', [])
            arrows = geo_result.get('arrows', [])
            lt = geo_result.get('line_types', {})
            dim_structs = geo_result.get('dimension_structures', [])
            contours = geo_result.get('contours', [])
            parts.append(f"直线: {len(lines)}条")
            parts.append(f"圆: {len(circles)}个")
            parts.append(f"箭头: {len(arrows)}个")
            parts.append(f"尺寸线对: {len(dim_structs)}对")
            parts.append(f"轮廓形状: {len(contours)}个")
            parts.append(f"线型: 实线{lt.get('solid_count',0)}/虚线{lt.get('dashed_count',0)}/点画线{lt.get('center_line_count',0)}")
            geo_summary = '；'.join(parts)

        metrics = result.get('metrics', {})
        rl_session_id = f"{os.path.basename(filepath)}_{int(time.time())}"
        rl_stats = metrics.get('rl_stats', {})

        v2_info = {
            'version': 'v2',
            'phases': OrchestratorV2.PHASES,
            'vlm_judge_enabled': VLM_JUDGE_ENABLED,
            'atlas_pack_enabled': ENABLE_ATLAS_PACK,
            'experience_store_dir': EXPERIENCE_STORE_DIR,
        }

        contract_details = _sanitize_for_json({
            'contract_list': metrics.get('contract_list', []),
            'contract_stats': metrics.get('contract_stats', {}),
        })

        phase_timings = metrics.get('phase_timings', {})

        judge_result = result.get('judge_result', {})
        rollback_stats = metrics.get('rollback_stats', {})
        experience_stats = metrics.get('experience_stats', {})

        return render_template('result_v2.html',
                           filename=filename,
                           ocr_results=ocr_results,
                           detection_results=detection_results,
                           errors=errors,
                           feedback=feedback,
                           gb_knowledge=gb_knowledge,
                           api_result=api_result,
                           report=report,
                           geo_summary=geo_summary,
                           rl_session_id=rl_session_id,
                           rl_stats=rl_stats,
                           v2_info=v2_info,
                           contract_details=contract_details,
                           phase_timings=phase_timings,
                           judge_result=judge_result,
                           rollback_stats=rollback_stats,
                           experience_stats=experience_stats)

    return redirect(request.url)

@app.route('/api/v2/rl_feedback', methods=['POST'])
def rl_feedback_v2():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No JSON data'}), 400

    session_id = data.get('session_id', '')
    error_description = data.get('error_description', '')
    feedback_type = data.get('feedback_type', '')

    if feedback_type not in ('confirmed', 'ignored', 'dismissed_all', 'partial_confirm', 'useful_guidance'):
        return jsonify({'success': False, 'error': f'Invalid feedback_type: {feedback_type}'}), 400

    if not session_id:
        return jsonify({'success': False, 'error': 'Missing session_id'}), 400

    try:
        orchestrator_v2.rl_memory.submit_feedback(session_id, error_description, feedback_type)

        if feedback_type in ('confirmed', 'ignored'):
            try:
                orchestrator_v2.experience_store.store(
                    {'errors': [{'description': error_description, 'feedback_type': feedback_type}]},
                    feedback_types=[feedback_type]
                )
            except Exception as e:
                logger.warning(f"[V2 RL Feedback] Experience store failed: {e}")

        stats = orchestrator_v2.rl_memory.get_stats()
        exp_stats = orchestrator_v2.experience_store.get_stats()
        return jsonify({
            'success': True,
            'message': f'Feedback "{feedback_type}" recorded for session {session_id}',
            'rl_stats': stats,
            'experience_stats': exp_stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v2/rl_stats')
def rl_stats_v2():
    rl_stats = orchestrator_v2.rl_memory.get_stats()
    exp_stats = orchestrator_v2.experience_store.get_stats()
    phase_timings = getattr(orchestrator_v2, '_phase_timings', {})
    return jsonify(_sanitize_for_json({
        'rl_stats': rl_stats,
        'experience_stats': exp_stats,
        'last_phase_timings': phase_timings,
        'vlm_judge_available': orchestrator_v2.vlm_judge.is_available(),
        'atlas_cases_count': len(orchestrator_v2._atlas_registry.get_all_cases()),
        'atlas_rules_count': len(orchestrator_v2._atlas_registry.get_all_rules()),
    }))

@app.route('/api/v2/experience_stats')
def experience_stats():
    stats = orchestrator_v2.experience_store.get_stats()
    return jsonify(_sanitize_for_json(stats))

@app.route('/api/atlas/case/<case_id>')
def atlas_case_detail(case_id):
    case = orchestrator_v2._atlas_registry.get_case(case_id)
    if case is None:
        return jsonify({'error': 'Case not found'}), 404
    return jsonify(_sanitize_for_json(case.to_dict()))

@app.route('/api/atlas/rules')
def atlas_rules_list():
    rules = orchestrator_v2._atlas_registry.get_all_rules()
    category = request.args.get('category', '')
    if category:
        rules = [r for r in rules if r.category == category]
    return json.dumps([_sanitize_for_json(r.to_dict()) for r in rules],
                      ensure_ascii=False, cls=NumpyEncoder)

@app.route('/api/atlas/feedback', methods=['POST'])
def atlas_feedback():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No JSON data'}), 400

    rule_id = data.get('rule_id', '')
    confirmed = data.get('confirmed', True)
    case_id = data.get('case_id', '')

    if not rule_id and not case_id:
        return jsonify({'success': False, 'error': 'Missing rule_id or case_id'}), 400

    try:
        if rule_id:
            orchestrator_v2._atlas_context_builder.rule_pack.record_feedback(rule_id, confirmed)

        if case_id:
            case = orchestrator_v2._atlas_registry.get_case(case_id)
            if case:
                case.add_feedback('confirmed' if confirmed else 'dismissed')

        return jsonify({
            'success': True,
            'message': f'Atlas feedback recorded: {"confirmed" if confirmed else "dismissed"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/atlas/stats')
def atlas_stats():
    registry_stats = orchestrator_v2._atlas_registry.get_stats()
    rule_pack_stats = orchestrator_v2._atlas_context_builder.rule_pack.get_stats()
    return jsonify(_sanitize_for_json({
        'registry': registry_stats,
        'rule_pack': rule_pack_stats,
        'enabled': ENABLE_ATLAS_PACK,
        'show_reference_in_ui': ATLAS_SHOW_REFERENCE_IN_UI,
    }))

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
