package dev.netzero.qidk;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Base64;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Net-Zero NPU demo: the QIDK dashboard as an Android app.
 *
 * The web UI and the recorded NPU/FP32 data are bundled in assets/www and served from a
 * private https origin, so the replay, day-ahead and pipeline views work offline.
 * Calls to api/... are forwarded to the nz_dashboard server (default http://127.0.0.1:8080,
 * i.e. the server on the same QIDK), which runs windows live on the Hexagon NPU.
 */
public class MainActivity extends Activity {
    private static final String HOST = "appassets.androidplatform.net";
    private static final String DEFAULT_SERVER = "http://127.0.0.1:8080";
    private static final int PICK_FILE = 42;

    private WebView web;
    private SharedPreferences prefs;
    private ValueCallback<Uri[]> fileCallback;
    private final ExecutorService pool = Executors.newCachedThreadPool();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        prefs = getSharedPreferences("nz", MODE_PRIVATE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        WebView.setWebContentsDebuggingEnabled(true);   // chrome://inspect for debugging the demo
        web = new WebView(this);
        web.setBackgroundColor(0xff0f1419);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(true);
        web.addJavascriptInterface(new Bridge(), "NZ");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if (!HOST.equals(u.getHost())) return null;
                String path = u.getPath() == null ? "" : u.getPath().replaceFirst("^/+", "");
                if (path.isEmpty()) path = "index.html";
                if (path.contains("..")) return notFound();
                try {
                    InputStream in = getAssets().open("www/" + path);
                    return new WebResourceResponse(mime(path), "UTF-8", in);
                } catch (IOException e) {
                    return notFound();
                }
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                return !HOST.equals(req.getUrl().getHost());   // stay inside the app
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams p) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = cb;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                try {
                    startActivityForResult(Intent.createChooser(i, "Choose a CSV window"), PICK_FILE);
                } catch (Exception e) {
                    fileCallback = null;
                    return false;
                }
                return true;
            }
        });
        setContentView(web);
        goImmersive();
        if (state == null) web.loadUrl("https://" + HOST + "/index.html");
        else web.restoreState(state);
    }

    private void goImmersive() {
        if (Build.VERSION.SDK_INT >= 30) {
            WindowInsetsController c = getWindow().getInsetsController();
            if (c != null) {
                c.hide(WindowInsets.Type.systemBars());
                c.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        } else {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) goImmersive();
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    protected void onActivityResult(int req, int result, Intent data) {
        if (req == PICK_FILE && fileCallback != null) {
            fileCallback.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result, data));
            fileCallback = null;
            return;
        }
        super.onActivityResult(req, result, data);
    }

    @Override
    public void onBackPressed() {
        // Back returns to the first tab instead of closing the demo mid-presentation.
        web.evaluateJavascript("window.__nzBack ? window.__nzBack() : false", (v) -> {
            if (!"true".equals(v)) MainActivity.super.onBackPressed();
        });
    }

    @Override
    protected void onDestroy() {
        pool.shutdownNow();
        web.destroy();
        super.onDestroy();
    }

    private String server() {
        return prefs.getString("server", DEFAULT_SERVER).replaceAll("/+$", "");
    }

    private static WebResourceResponse notFound() {
        WebResourceResponse r = new WebResourceResponse("text/plain", "UTF-8",
                new ByteArrayInputStream("not found".getBytes(StandardCharsets.UTF_8)));
        r.setStatusCodeAndReasonPhrase(404, "Not Found");
        return r;
    }

    private static String mime(String p) {
        if (p.endsWith(".html")) return "text/html";
        if (p.endsWith(".js")) return "text/javascript";
        if (p.endsWith(".css")) return "text/css";
        if (p.endsWith(".json")) return "application/json";
        if (p.endsWith(".svg")) return "image/svg+xml";
        if (p.endsWith(".png")) return "image/png";
        return "application/octet-stream";
    }

    private static byte[] readAll(InputStream in) throws IOException {
        if (in == null) return new byte[0];
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] buf = new byte[16384];
        int n;
        while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
        in.close();
        return bo.toByteArray();
    }

    private void js(String code) {
        runOnUiThread(() -> web.evaluateJavascript(code, null));
    }

    /** Methods callable from the page as window.NZ.* */
    private class Bridge {
        /** HTTP request to the NPU server; the answer arrives via window.__nzDone(id, status, text). */
        @JavascriptInterface
        public void request(final int id, final String method, final String path, final String b64Body) {
            pool.execute(() -> {
                int status;
                String text;
                String base = server();
                try {
                    HttpURLConnection c = (HttpURLConnection) new URL(base + "/" + path).openConnection();
                    c.setConnectTimeout(3000);
                    c.setReadTimeout(120000);
                    c.setRequestMethod(method);
                    if (b64Body != null && !b64Body.isEmpty()) {
                        byte[] body = Base64.decode(b64Body, Base64.DEFAULT);
                        c.setDoOutput(true);
                        c.setRequestProperty("Content-Type", "application/octet-stream");
                        c.setFixedLengthStreamingMode(body.length);
                        try (OutputStream o = c.getOutputStream()) { o.write(body); }
                    }
                    status = c.getResponseCode();
                    InputStream in = status >= 400 ? c.getErrorStream() : c.getInputStream();
                    text = new String(readAll(in), StandardCharsets.UTF_8);
                    c.disconnect();
                } catch (Exception e) {
                    status = 0;
                    text = "NPU server not reachable at " + base
                            + " (start it with scripts/14_dashboard_start.sh, or tap ⚙ to set the QIDK address)";
                }
                js("window.__nzDone(" + id + "," + status + "," + JSONObject.quote(text) + ")");
            });
        }

        @JavascriptInterface
        public String getServer() { return server(); }

        @JavascriptInterface
        public String appDevice() {
            String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : Build.HARDWARE;
            return Build.MODEL + " · " + soc + " · Android " + Build.VERSION.RELEASE;
        }

        @JavascriptInterface
        public void openSettings() {
            runOnUiThread(() -> {
                final EditText in = new EditText(MainActivity.this);
                in.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
                in.setText(server());
                in.setSelectAllOnFocus(true);
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("NPU server")
                        .setMessage("Address of nz_dashboard on the QIDK.\n"
                                + "On the QIDK itself: " + DEFAULT_SERVER + "\n"
                                + "From another phone: http://<QIDK Wi-Fi IP>:8080")
                        .setView(in)
                        .setPositiveButton("Save", (d, w) -> {
                            String v = in.getText().toString().trim();
                            if (!v.startsWith("http")) v = "http://" + v;
                            prefs.edit().putString("server", v).apply();
                            js("window.__nzServerChanged && window.__nzServerChanged()");
                        })
                        .setNeutralButton("Default", (d, w) -> {
                            prefs.edit().remove("server").apply();
                            js("window.__nzServerChanged && window.__nzServerChanged()");
                        })
                        .setNegativeButton("Cancel", null)
                        .show();
            });
        }

        /** Saves a text file (CSV template) to Downloads. */
        @JavascriptInterface
        public void saveFile(final String name, final String text) {
            pool.execute(() -> {
                String msg;
                try {
                    byte[] data = text.getBytes(StandardCharsets.UTF_8);
                    if (Build.VERSION.SDK_INT >= 29) {
                        ContentValues v = new ContentValues();
                        v.put(MediaStore.Downloads.DISPLAY_NAME, name);
                        v.put(MediaStore.Downloads.MIME_TYPE, "text/csv");
                        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
                        try (OutputStream o = getContentResolver().openOutputStream(uri)) { o.write(data); }
                        msg = "Saved to Downloads/" + name;
                    } else {
                        File f = new File(getExternalFilesDir(null), name);
                        try (FileOutputStream o = new FileOutputStream(f)) { o.write(data); }
                        msg = "Saved to " + f.getAbsolutePath();
                    }
                } catch (Exception e) {
                    msg = "Could not save: " + e.getMessage();
                }
                final String m = msg;
                runOnUiThread(() -> Toast.makeText(MainActivity.this, m, Toast.LENGTH_LONG).show());
            });
        }
    }
}
