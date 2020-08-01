
class CorsMiddleware(object):
    def process_response(self, req, resp):
        req["Access-Control-Allow-Origin"] = "*"
        return resp
