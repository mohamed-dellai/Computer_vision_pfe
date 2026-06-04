from flask import Blueprint, jsonify, request
import os
from werkzeug.utils import secure_filename
from services.model_optimizer import ModelOptimizer, normalize_export_target

def create_models_blueprint(models_dir, allowed_extensions, available_models_func):
    bp = Blueprint('models', __name__, url_prefix='/models')

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @bp.route('', methods=['GET'])
    def list_models():
        return jsonify({'models': available_models_func()})

    @bp.route('/upload', methods=['POST'])
    def upload_model():
        if 'file' not in request.files:
            return jsonify({'error': 'no file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        if not allowed_file(file.filename):
            allowed = ', '.join(f'.{ext}' for ext in sorted(allowed_extensions))
            return jsonify({'error': f'only {allowed} files allowed'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(models_dir, filename)
        file.save(filepath)
        return jsonify({'message': 'model uploaded', 'filename': filename}), 201

    @bp.route('/optimize', methods=['POST'])
    def optimize_model():
        data = request.json or {}
        source = data.get('source')
        try:
            target = normalize_export_target(data.get('target') or 'onnx')
            imgsz = _parse_imgsz(data.get('imgsz', 640))
            opset = _parse_int(data.get('opset', 12), 'opset', minimum=7)
            dynamic = _parse_bool(data.get('dynamic', False))
            half = _parse_bool(data.get('half', False))
            simplify = _parse_bool(data.get('simplify', False))
            device = data.get('device')
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        if not source:
            return jsonify({'error': 'source field required'}), 400

        try:
            optimizer = ModelOptimizer(models_dir)
            result = optimizer.optimize(
                source,
                target,
                imgsz=imgsz,
                opset=opset,
                dynamic=dynamic,
                half=half,
                simplify=simplify,
                device=device,
            )
            return jsonify({
                'message': 'model exported',
                'filename': result['filename'],
                'format': result['format'],
                'format_label': result['format_label'],
            }), 200
        except (FileNotFoundError, ValueError) as e:
            return jsonify({'error': str(e)}), 400
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            return jsonify({'error': f'export failed: {e}'}), 500

    return bp


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _parse_int(value, name, minimum=None):
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return parsed


def _parse_imgsz(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError('imgsz must be an integer or [height, width]')
        return [_parse_int(value[0], 'imgsz height', minimum=32), _parse_int(value[1], 'imgsz width', minimum=32)]
    return _parse_int(value, 'imgsz', minimum=32)
