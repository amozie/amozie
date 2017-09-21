class Technique():
    def __init__(self):
        self.techniques = []

    def run(self, data) -> list:
        ret = self.techniques
        self.techniques = []
        return ret

    def _add_technique(self, name, value, row=0, style='', width=None, alpha=None, x_axis=None):
        self.techniques.append(
            {
                'name': name,
                'value': value,
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis
            }
        )
