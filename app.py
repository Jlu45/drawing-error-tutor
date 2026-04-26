from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import json
from src.rag_knowledge_base import DualKnowledgeBase
from src.multi_agent_system import DrawingOrchestrator
from config_loader import (
    MULTIMODAL_API_URL, MULTIMODAL_API_KEY, LLM_MODEL,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
)

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
    return suggestions

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

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
