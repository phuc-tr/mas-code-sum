from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base-multi-sum')

    text = """Summarize python: def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG data.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image"""

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=20)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    # this prints: "Convert a SVG string to a QImage."
