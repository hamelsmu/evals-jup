"use strict";
(self["webpackChunkjupyvibe"] = self["webpackChunkjupyvibe"] || []).push([["lib_index_js"],{

/***/ "./lib/cellTypeSwitcher.js"
/*!*********************************!*\
  !*** ./lib/cellTypeSwitcher.js ***!
  \*********************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   CustomCellTypeSwitcher: () => (/* binding */ CustomCellTypeSwitcher)
/* harmony export */ });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "webpack/sharing/consume/default/react");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _jupyterlab_translation__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @jupyterlab/translation */ "webpack/sharing/consume/default/@jupyterlab/translation");
/* harmony import */ var _jupyterlab_translation__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_translation__WEBPACK_IMPORTED_MODULE_3__);
/**
 * Custom cell type switcher that adds "Prompt" option to the dropdown.
 *
 * Replaces the default JupyterLab cell type switcher via IToolbarWidgetRegistry.
 */




const TOOLBAR_CELLTYPE_CLASS = 'jp-Notebook-toolbarCellType';
const TOOLBAR_CELLTYPE_DROPDOWN_CLASS = 'jp-Notebook-toolbarCellTypeDropdown';
const PROMPT_METADATA_KEY = 'jupyvibe';
const PROMPT_CELL_CLASS = 'jupyvibe-prompt-cell';
/**
 * Custom cell type switcher widget with Prompt option.
 */
class CustomCellTypeSwitcher extends _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__.ReactWidget {
    constructor(panel, _promptCellManager, translator) {
        super();
        /**
         * Handle cell type change from dropdown.
         */
        this.handleChange = (event) => {
            const newType = event.target.value;
            if (newType === '-') {
                return;
            }
            if (newType === 'prompt') {
                this._convertToPrompt();
            }
            else {
                // Remove prompt metadata if converting away from prompt
                this._removePromptMetadata();
                _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__.NotebookActions.changeCellType(this._notebook, newType);
            }
            this._notebook.activate();
        };
        /**
         * Handle keyboard navigation.
         */
        this.handleKeyDown = (event) => {
            if (event.key === 'Enter') {
                this._notebook.activate();
            }
        };
        this._trans = (translator ?? _jupyterlab_translation__WEBPACK_IMPORTED_MODULE_3__.nullTranslator).load('jupyterlab');
        this.addClass(TOOLBAR_CELLTYPE_CLASS);
        this._notebook = panel.content;
        if (this._notebook.model) {
            this.update();
        }
        this._notebook.activeCellChanged.connect(this.update, this);
        this._notebook.selectionChanged.connect(this.update, this);
        // Clean up on dispose
        panel.disposed.connect(() => {
            this._notebook.activeCellChanged.disconnect(this.update, this);
            this._notebook.selectionChanged.disconnect(this.update, this);
        });
    }
    /**
     * Convert selected cells to prompt type.
     */
    _convertToPrompt() {
        const notebook = this._notebook;
        if (!notebook.model) {
            return;
        }
        // Collect indices of cells to convert (don't modify during iteration)
        const indicesToConvert = [];
        notebook.widgets.forEach((cell, index) => {
            if (!notebook.isSelectedOrActive(cell)) {
                return;
            }
            // Check if already a prompt cell
            const metadata = cell.model.getMetadata(PROMPT_METADATA_KEY);
            if (metadata?.isPromptCell) {
                return;
            }
            indicesToConvert.push(index);
        });
        // Convert cells (process in reverse to preserve indices)
        for (let i = indicesToConvert.length - 1; i >= 0; i--) {
            const index = indicesToConvert[i];
            const cell = notebook.widgets[index];
            if (!cell)
                continue;
            const needsTypeChange = cell.model.type !== 'markdown';
            // First change the cell type if needed (using NotebookActions for proper handling)
            if (needsTypeChange) {
                // Make this cell the active cell for NotebookActions
                notebook.activeCellIndex = index;
                notebook.deselectAll();
                _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__.NotebookActions.changeCellType(notebook, 'markdown');
            }
            // Now set the prompt metadata on the (possibly new) cell
            const targetCell = notebook.widgets[index];
            if (targetCell) {
                targetCell.model.setMetadata(PROMPT_METADATA_KEY, {
                    isPromptCell: true,
                    model: 'claude-sonnet-4-20250514'
                });
                // Add prompt prefix if source is empty
                const source = targetCell.model.sharedModel.getSource();
                if (!source || !source.trim()) {
                    targetCell.model.sharedModel.setSource('**AI Prompt:** ');
                }
                // Add styling class
                if (!targetCell.hasClass(PROMPT_CELL_CLASS)) {
                    targetCell.addClass(PROMPT_CELL_CLASS);
                }
            }
        }
        notebook.deselectAll();
    }
    /**
     * Remove prompt metadata from selected cells.
     */
    _removePromptMetadata() {
        const notebook = this._notebook;
        notebook.widgets.forEach((cell) => {
            if (!notebook.isSelectedOrActive(cell)) {
                return;
            }
            // Remove prompt metadata
            const metadata = cell.model.getMetadata(PROMPT_METADATA_KEY);
            if (metadata?.isPromptCell) {
                cell.model.deleteMetadata(PROMPT_METADATA_KEY);
                cell.removeClass(PROMPT_CELL_CLASS);
            }
        });
    }
    /**
     * Get the current cell type value for the dropdown.
     */
    _getValue() {
        const notebook = this._notebook;
        if (!notebook.activeCell) {
            return '-';
        }
        // Check if active cell is a prompt cell
        let value = this._isPromptCell(notebook.activeCell.model)
            ? 'prompt'
            : notebook.activeCell.model.type;
        // Check all selected cells for consistency
        for (const widget of notebook.widgets) {
            if (notebook.isSelectedOrActive(widget)) {
                const cellType = this._isPromptCell(widget.model) ? 'prompt' : widget.model.type;
                if (cellType !== value) {
                    return '-';
                }
            }
        }
        return value;
    }
    /**
     * Check if a cell model is a prompt cell.
     */
    _isPromptCell(model) {
        const metadata = model.getMetadata(PROMPT_METADATA_KEY);
        return metadata?.isPromptCell === true;
    }
    render() {
        const value = this._getValue();
        return (react__WEBPACK_IMPORTED_MODULE_0__.createElement("select", { className: TOOLBAR_CELLTYPE_DROPDOWN_CLASS, onChange: this.handleChange, onKeyDown: this.handleKeyDown, value: value, "aria-label": this._trans.__('Cell type'), title: this._trans.__('Select the cell type') },
            react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { value: "-" }, "-"),
            react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { value: "code" }, this._trans.__('Code')),
            react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { value: "markdown" }, this._trans.__('Markdown')),
            react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { value: "raw" }, this._trans.__('Raw')),
            react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { value: "prompt" }, this._trans.__('Prompt'))));
    }
}


/***/ },

/***/ "./lib/index.js"
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/mainmenu */ "webpack/sharing/consume/default/@jupyterlab/mainmenu");
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @jupyterlab/settingregistry */ "webpack/sharing/consume/default/@jupyterlab/settingregistry");
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _jupyterlab_translation__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @jupyterlab/translation */ "webpack/sharing/consume/default/@jupyterlab/translation");
/* harmony import */ var _jupyterlab_translation__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_translation__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _promptCell__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./promptCell */ "./lib/promptCell.js");
/* harmony import */ var _kernelConnector__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./kernelConnector */ "./lib/kernelConnector.js");
/* harmony import */ var _settings__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./settings */ "./lib/settings.js");
/* harmony import */ var _cellTypeSwitcher__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./cellTypeSwitcher */ "./lib/cellTypeSwitcher.js");
/* harmony import */ var _modelPicker__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ./modelPicker */ "./lib/modelPicker.js");
/* harmony import */ var _tokens__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./tokens */ "./lib/tokens.js");
/**
 * AI-powered Jupyter Lab extension with prompt cells.
 *
 * Features:
 * - $variable syntax to reference kernel variables in prompts
 * - &function syntax to give AI access to kernel functions as tools
 * - Prompt cells that see all preceding cells and kernel state
 */












const PLUGIN_ID = 'jupyvibe:plugin';
/**
 * Settings plugin that provides IExtensionSettings.
 */
const settingsPlugin = {
    id: 'jupyvibe:settings',
    description: 'Provides Jupyvibe extension settings',
    autoStart: true,
    requires: [],
    optional: [_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_3__.ISettingRegistry],
    provides: _tokens__WEBPACK_IMPORTED_MODULE_11__.IExtensionSettings,
    activate: async (app, settingRegistry) => {
        const settingsManager = new _settings__WEBPACK_IMPORTED_MODULE_8__.SettingsManager();
        if (settingRegistry) {
            await settingsManager.initialize(settingRegistry);
            console.log('[jupyvibe] Settings loaded:', settingsManager.toJSON());
        }
        else {
            console.log('[jupyvibe] No setting registry available, using defaults');
        }
        return settingsManager;
    }
};
/**
 * Kernel connector factory plugin.
 */
const kernelConnectorPlugin = {
    id: 'jupyvibe:kernel-connector',
    description: 'Provides kernel connector factory',
    autoStart: true,
    provides: _tokens__WEBPACK_IMPORTED_MODULE_11__.IKernelConnectorFactory,
    activate: () => {
        return {
            create: (sessionContext) => new _kernelConnector__WEBPACK_IMPORTED_MODULE_7__.KernelConnector(sessionContext)
        };
    }
};
/**
 * Prompt cell manager plugin.
 */
const promptCellManagerPlugin = {
    id: 'jupyvibe:prompt-cell-manager',
    description: 'Manages AI prompt cells',
    autoStart: true,
    requires: [_tokens__WEBPACK_IMPORTED_MODULE_11__.IExtensionSettings],
    provides: _tokens__WEBPACK_IMPORTED_MODULE_11__.IPromptCellManager,
    activate: (app, settings) => {
        const manager = new _promptCell__WEBPACK_IMPORTED_MODULE_6__.PromptCellManager();
        manager.setSettings(settings);
        return manager;
    }
};
/**
 * Main plugin that sets up commands and UI.
 */
const mainPlugin = {
    id: PLUGIN_ID,
    description: 'AI-powered prompt cells for JupyterLab',
    autoStart: true,
    requires: [_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.INotebookTracker, _tokens__WEBPACK_IMPORTED_MODULE_11__.IPromptCellManager, _tokens__WEBPACK_IMPORTED_MODULE_11__.IKernelConnectorFactory, _tokens__WEBPACK_IMPORTED_MODULE_11__.IExtensionSettings],
    optional: [_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ICommandPalette, _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__.IMainMenu],
    activate: (app, notebookTracker, promptCellManager, connectorFactory, settings, palette, mainMenu) => {
        console.log('Jupyvibe extension activated');
        // Command to insert a new prompt cell
        const insertPromptCommand = 'jupyvibe:insert-prompt-cell';
        app.commands.addCommand(insertPromptCommand, {
            label: 'Insert AI Prompt Cell',
            caption: 'Insert a new AI prompt cell below the current cell',
            execute: () => {
                const panel = notebookTracker.currentWidget;
                if (!panel) {
                    return;
                }
                promptCellManager.insertPromptCell(panel);
            }
        });
        // Command to run prompt cell
        const runPromptCommand = 'jupyvibe:run-prompt';
        app.commands.addCommand(runPromptCommand, {
            label: 'Run AI Prompt',
            caption: 'Execute the current prompt cell',
            execute: async () => {
                const panel = notebookTracker.currentWidget;
                if (!panel) {
                    return;
                }
                await promptCellManager.executePromptCell(panel);
            }
        });
        // Add keyboard shortcuts
        app.commands.addKeyBinding({
            command: insertPromptCommand,
            keys: ['Accel Shift P'],
            selector: '.jp-Notebook'
        });
        // "P" in command mode inserts prompt cell (like "M" for markdown, "Y" for code)
        app.commands.addKeyBinding({
            command: insertPromptCommand,
            keys: ['P'],
            selector: '.jp-Notebook.jp-mod-commandMode:not(.jp-mod-readWrite) :focus'
        });
        // Shift+Enter on prompt cells runs AI instead of normal execution
        app.commands.addKeyBinding({
            command: runPromptCommand,
            keys: ['Shift Enter'],
            selector: '.jp-Notebook.jp-mod-editMode .jp-Cell.jupyvibe-prompt-cell'
        });
        app.commands.addKeyBinding({
            command: runPromptCommand,
            keys: ['Shift Enter'],
            selector: '.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected.jupyvibe-prompt-cell'
        });
        // Add to command palette
        if (palette) {
            palette.addItem({
                command: insertPromptCommand,
                category: 'AI'
            });
            palette.addItem({
                command: runPromptCommand,
                category: 'AI'
            });
        }
        // Add to Edit menu
        if (mainMenu) {
            mainMenu.editMenu.addGroup([
                { command: insertPromptCommand },
                { command: runPromptCommand }
            ], 20);
        }
        // Helper to set up a notebook panel
        const setupPanel = (panel) => {
            const doSetup = () => {
                // Skip if notebook was closed before context became ready
                if (panel.isDisposed) {
                    return;
                }
                // Add toolbar button for inserting prompt cells
                const button = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ToolbarButton({
                    icon: _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_5__.addIcon,
                    onClick: () => {
                        promptCellManager.insertPromptCell(panel);
                    },
                    tooltip: 'Insert AI Prompt Cell (Cmd/Ctrl+Shift+P)',
                    label: 'AI Prompt'
                });
                panel.toolbar.insertAfter('cellType', 'jupyvibe-insert', button);
                // Add model picker to toolbar
                const modelPicker = new _modelPicker__WEBPACK_IMPORTED_MODULE_10__.ModelPickerWidget(settings);
                panel.toolbar.insertAfter('jupyvibe-insert', 'jupyvibe-model-picker', modelPicker);
                // Use requestAnimationFrame to wait for cells to be rendered
                requestAnimationFrame(() => {
                    if (panel.isDisposed) {
                        return;
                    }
                    const connector = connectorFactory.create(panel.sessionContext);
                    promptCellManager.setupNotebook(panel, connector);
                });
            };
            if (panel.context.isReady) {
                doSetup();
            }
            else {
                panel.context.ready.then(doSetup);
            }
        };
        // Track new notebooks
        notebookTracker.widgetAdded.connect((_, panel) => setupPanel(panel));
        // Process existing notebooks
        notebookTracker.forEach(setupPanel);
    }
};
/**
 * Plugin that replaces the cell type dropdown with one that includes "Prompt".
 */
const cellTypeSwitcherPlugin = {
    id: 'jupyvibe:cell-type-switcher',
    description: 'Adds Prompt option to cell type dropdown',
    autoStart: true,
    requires: [_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.IToolbarWidgetRegistry, _tokens__WEBPACK_IMPORTED_MODULE_11__.IPromptCellManager, _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.INotebookWidgetFactory],
    optional: [_jupyterlab_translation__WEBPACK_IMPORTED_MODULE_4__.ITranslator],
    activate: (app, toolbarRegistry, promptCellManager, _notebookWidgetFactory, translator) => {
        console.log('[jupyvibe] Registering custom cell type switcher (after notebook widget factory)');
        const oldFactory = toolbarRegistry.addFactory('Notebook', 'cellType', (panel) => {
            console.log('[jupyvibe] Creating CustomCellTypeSwitcher for panel:', panel.id);
            return new _cellTypeSwitcher__WEBPACK_IMPORTED_MODULE_9__.CustomCellTypeSwitcher(panel, promptCellManager, translator ?? undefined);
        });
        console.log('[jupyvibe] Replaced cellType factory, old factory was:', oldFactory ? 'present' : 'none');
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = ([settingsPlugin, kernelConnectorPlugin, promptCellManagerPlugin, cellTypeSwitcherPlugin, mainPlugin]);


/***/ },

/***/ "./lib/kernelConnector.js"
/*!********************************!*\
  !*** ./lib/kernelConnector.js ***!
  \********************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   KernelConnector: () => (/* binding */ KernelConnector),
/* harmony export */   KernelConnectorFactory: () => (/* binding */ KernelConnectorFactory)
/* harmony export */ });
/**
 * Kernel connector for variable and function introspection.
 */
/**
 * Connects to a Jupyter kernel to introspect variables and functions.
 * Implements IKernelConnector for dependency injection.
 */
class KernelConnector {
    constructor(session) {
        this._session = session;
    }
    /**
     * Check if the kernel is available.
     */
    get kernelAvailable() {
        return !!this._session.session?.kernel;
    }
    /**
     * Execute code silently and capture output.
     */
    async execute(code, onOutput) {
        const kernel = this._session.session?.kernel;
        if (!kernel) {
            return null;
        }
        const content = {
            code,
            stop_on_error: false,
            store_history: false,
            silent: true
        };
        const future = kernel.requestExecute(content);
        if (onOutput) {
            future.onIOPub = onOutput;
        }
        try {
            return (await future.done);
        }
        finally {
            future.dispose();
        }
    }
    /**
     * Execute code and return stdout output.
     */
    async executeAndCapture(code) {
        let output = '';
        await this.execute(code, (msg) => {
            const msgType = msg.header.msg_type;
            const content = msg.content;
            if (msgType === 'stream' && content.name === 'stdout') {
                output += content.text;
            }
            else if (msgType === 'execute_result') {
                const data = content.data;
                if (data && data['text/plain']) {
                    output += data['text/plain'];
                }
            }
        });
        return output.trim();
    }
    /**
     * Get the value of a variable by name.
     */
    async getVariable(name) {
        const code = `
import json as _json_mod
try:
    _var = ${name}
    _result = {
        "name": "${name}",
        "type": type(_var).__name__,
        "repr": repr(_var)[:500]
    }
    print(_json_mod.dumps(_result))
    del _var, _result
except Exception as _e:
    print(_json_mod.dumps({"error": str(_e)}))
`;
        try {
            const output = await this.executeAndCapture(code);
            if (!output) {
                return null;
            }
            const result = JSON.parse(output);
            if (result.error) {
                console.warn(`Error getting variable ${name}:`, result.error);
                return null;
            }
            return result;
        }
        catch (e) {
            console.error(`Failed to get variable ${name}:`, e);
            return null;
        }
    }
    /**
     * Get information about a function.
     * Parses numpy/Google-style docstrings for parameter descriptions.
     */
    async getFunction(name) {
        const code = `
import json as _json_mod
import inspect as _inspect_mod
import re as _re_mod
try:
    _func = ${name}
    if not callable(_func):
        print(_json_mod.dumps({"error": "Not callable"}))
    else:
        _sig = str(_inspect_mod.signature(_func))
        _doc = _inspect_mod.getdoc(_func) or "No documentation"
        
        # Parse docstring for parameter descriptions (numpy/Google style)
        _param_docs = {}
        try:
            _lines = _doc.splitlines()
            _in_params_section = False
            _current_param = None
            _current_desc = []
            
            for _line in _lines:
                _stripped = _line.strip()
                _lower = _stripped.lower()
                
                # Detect section headers
                if _lower in ('parameters', 'args', 'arguments', 'params'):
                    _in_params_section = True
                    continue
                elif _lower in ('returns', 'return', 'raises', 'examples', 'notes', 'see also', 'attributes'):
                    # End of parameters section
                    if _current_param and _current_desc:
                        _param_docs[_current_param] = ' '.join(_current_desc).strip()
                    _in_params_section = False
                    _current_param = None
                    _current_desc = []
                    continue
                
                if not _in_params_section:
                    continue
                
                # Skip section underlines (numpy style)
                if _stripped and all(c == '-' for c in _stripped):
                    continue
                
                # Check if this is a new parameter line
                # Numpy style: "param : type" or "param: type"
                # Google style: "param (type): description" or "param: description"
                _param_match = _re_mod.match(r'^(\\w+)\\s*(?:\\(.*?\\))?\\s*:(.*)$', _stripped)
                if _param_match and not _line.startswith(' ' * 4) or (_param_match and _line and _line[0] not in ' \\t'):
                    # Save previous param
                    if _current_param and _current_desc:
                        _param_docs[_current_param] = ' '.join(_current_desc).strip()
                    
                    _current_param = _param_match.group(1)
                    _rest = _param_match.group(2).strip()
                    _current_desc = [_rest] if _rest else []
                elif _current_param and _stripped:
                    # Continuation line
                    _current_desc.append(_stripped)
            
            # Save last param
            if _current_param and _current_desc:
                _param_docs[_current_param] = ' '.join(_current_desc).strip()
        except (AttributeError, TypeError, ValueError) as _parse_err:
            # Docstring parsing is best-effort; fall back to empty on parse failures
            _param_docs = {}
        
        _params = {}
        for _pname, _param in _inspect_mod.signature(_func).parameters.items():
            # Use parsed docstring description if available, otherwise use param name
            _desc = _param_docs.get(_pname, _pname)
            _pinfo = {"type": "string", "description": _desc}
            if _param.annotation != _inspect_mod.Parameter.empty:
                _ann = _param.annotation
                if hasattr(_ann, '__name__'):
                    _pinfo["type"] = _ann.__name__
                elif hasattr(_ann, '__origin__'):
                    _pinfo["type"] = str(_ann)
            if _param.default != _inspect_mod.Parameter.empty:
                _pinfo["default"] = repr(_param.default)
            _params[_pname] = _pinfo
        # Extract return type annotation
        _return_type = None
        _ret_ann = _inspect_mod.signature(_func).return_annotation
        if _ret_ann != _inspect_mod.Parameter.empty:
            if hasattr(_ret_ann, '__name__'):
                _return_type = _ret_ann.__name__
            elif hasattr(_ret_ann, '__origin__'):
                _return_type = str(_ret_ann)
            else:
                _return_type = str(_ret_ann)
        
        # Append return type to docstring (like toolslm pattern)
        _full_doc = _doc[:500]
        if _return_type:
            _full_doc += f"\\n\\nReturns:\\n- type: {_return_type}"
        
        _result = {
            "name": "${name}",
            "signature": _sig,
            "docstring": _full_doc,
            "parameters": _params,
            "return_type": _return_type
        }
        print(_json_mod.dumps(_result))
        del _func, _sig, _doc, _params, _result, _param_docs
except Exception as _e:
    print(_json_mod.dumps({"error": str(_e)}))
`;
        try {
            const output = await this.executeAndCapture(code);
            if (!output) {
                return null;
            }
            const result = JSON.parse(output);
            if (result.error) {
                console.warn(`Error getting function ${name}:`, result.error);
                return null;
            }
            return result;
        }
        catch (e) {
            console.error(`Failed to get function ${name}:`, e);
            return null;
        }
    }
}
/**
 * Factory for creating KernelConnector instances.
 * Implements IKernelConnectorFactory for dependency injection.
 */
class KernelConnectorFactory {
    create(sessionContext) {
        return new KernelConnector(sessionContext);
    }
}


/***/ },

/***/ "./lib/modelPicker.js"
/*!****************************!*\
  !*** ./lib/modelPicker.js ***!
  \****************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   ModelPickerWidget: () => (/* binding */ ModelPickerWidget)
/* harmony export */ });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "webpack/sharing/consume/default/react");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/coreutils */ "webpack/sharing/consume/default/@jupyterlab/coreutils");
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _jupyterlab_services__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @jupyterlab/services */ "webpack/sharing/consume/default/@jupyterlab/services");
/* harmony import */ var _jupyterlab_services__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_services__WEBPACK_IMPORTED_MODULE_3__);
/**
 * Model picker component for selecting AI provider and model.
 */





const PROVIDER_ORDER = ['anthropic', 'openai', 'gemini'];
function ModelPickerComponent({ settings }) {
    const [providers, setProviders] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)({});
    const [modelsByProvider, setModelsByProvider] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)({});
    const [selectedProvider, setSelectedProvider] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)(settings.provider || 'anthropic');
    const [selectedModel, setSelectedModel] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)(settings.defaultModel || '');
    const [loading, setLoading] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)(true);
    const [error, setError] = (0,react__WEBPACK_IMPORTED_MODULE_0__.useState)(null);
    (0,react__WEBPACK_IMPORTED_MODULE_0__.useEffect)(() => {
        const fetchModels = async () => {
            try {
                setLoading(true);
                setError(null);
                const serverSettings = _jupyterlab_services__WEBPACK_IMPORTED_MODULE_3__.ServerConnection.makeSettings();
                const url = _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_2__.URLExt.join(serverSettings.baseUrl, 'jupyvibe', 'models');
                const response = await _jupyterlab_services__WEBPACK_IMPORTED_MODULE_3__.ServerConnection.makeRequest(url, {}, serverSettings);
                if (!response.ok) {
                    throw new Error(`Failed to fetch models: ${response.status}`);
                }
                const data = await response.json();
                setProviders(data.providers);
                setModelsByProvider(data.models);
                // If current model isn't in the list for the provider, select first available
                const providerModels = data.models[settings.provider] || [];
                if (providerModels.length > 0 && !providerModels.find(m => m.id === settings.defaultModel)) {
                    setSelectedModel(providerModels[0].id);
                }
            }
            catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load models');
            }
            finally {
                setLoading(false);
            }
        };
        fetchModels();
    }, [settings.provider, settings.defaultModel]);
    const handleProviderChange = async (e) => {
        const newProvider = e.target.value;
        setSelectedProvider(newProvider);
        // Select first model for new provider
        const providerModels = modelsByProvider[newProvider] || [];
        const newModel = providerModels.length > 0 ? providerModels[0].id : '';
        setSelectedModel(newModel);
        // Save to settings
        if (settings.set) {
            await settings.set('provider', newProvider);
            if (newModel) {
                await settings.set('defaultModel', newModel);
            }
        }
    };
    const handleModelChange = async (e) => {
        const newModel = e.target.value;
        setSelectedModel(newModel);
        // Save to settings
        if (settings.set) {
            await settings.set('defaultModel', newModel);
        }
    };
    if (loading) {
        return react__WEBPACK_IMPORTED_MODULE_0__.createElement("span", { className: "jupyvibe-model-picker-loading" }, "Loading...");
    }
    if (error) {
        return (react__WEBPACK_IMPORTED_MODULE_0__.createElement("span", { className: "jupyvibe-model-picker-error", title: error }, "\u26A0\uFE0F Error"));
    }
    const currentModels = modelsByProvider[selectedProvider] || [];
    return (react__WEBPACK_IMPORTED_MODULE_0__.createElement("div", { className: "jupyvibe-model-picker" },
        react__WEBPACK_IMPORTED_MODULE_0__.createElement("select", { className: "jupyvibe-provider-select", value: selectedProvider, onChange: handleProviderChange, title: "Select AI Provider" }, PROVIDER_ORDER.filter(p => p in providers).map(providerId => (react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { key: providerId, value: providerId }, providers[providerId])))),
        react__WEBPACK_IMPORTED_MODULE_0__.createElement("select", { className: "jupyvibe-model-select", value: selectedModel, onChange: handleModelChange, title: "Select Model" }, currentModels.map(model => (react__WEBPACK_IMPORTED_MODULE_0__.createElement("option", { key: model.id, value: model.id }, model.name))))));
}
class ModelPickerWidget extends _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ReactWidget {
    constructor(settings) {
        super();
        this._settings = settings;
        this.addClass('jupyvibe-model-picker-widget');
    }
    render() {
        return react__WEBPACK_IMPORTED_MODULE_0__.createElement(ModelPickerComponent, { settings: this._settings });
    }
}


/***/ },

/***/ "./lib/promptCell.js"
/*!***************************!*\
  !*** ./lib/promptCell.js ***!
  \***************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   PromptCellManager: () => (/* binding */ PromptCellManager)
/* harmony export */ });
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/cells */ "webpack/sharing/consume/default/@jupyterlab/cells");
/* harmony import */ var _jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _promptParser__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./promptParser */ "./lib/promptParser.js");
/* harmony import */ var _promptModel__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./promptModel */ "./lib/promptModel.js");
/**
 * Prompt cell management and execution.
 */





/** Supported image MIME types for multimodal context */
const IMAGE_MIME_TYPES = ['image/png', 'image/jpeg', 'image/gif'];
/** MIME type patterns for declarative chart specs */
const VEGALITE_MIME_PATTERN = /^application\/vnd\.vegalite\.v\d+\+json$/;
const PLOTLY_MIME = 'application/vnd.plotly.v1+json';
const PROMPT_CELL_CLASS = 'jupyvibe-prompt-cell';
const PROMPT_OUTPUT_CLASS = 'jupyvibe-prompt-output';
const PROMPT_METADATA_KEY = 'jupyvibe';
/**
 * Manages prompt cells within notebooks.
 * Implements IPromptCellManager for dependency injection.
 */
class PromptCellManager {
    constructor() {
        this._connectors = new Map();
        this._settings = null;
    }
    /**
     * Set the settings instance.
     */
    setSettings(settings) {
        this._settings = settings;
    }
    /**
     * Set up a notebook for prompt cell handling.
     */
    setupNotebook(panel, connector) {
        const notebookId = panel.id;
        this._connectors.set(notebookId, connector);
        const notebook = panel.content;
        // Style all prompt and output cells - works with JupyterLab 4 windowing
        const stylePromptCells = () => {
            if (panel.isDisposed || !notebook.model) {
                return;
            }
            const cellCount = notebook.model.cells.length;
            for (let i = 0; i < cellCount; i++) {
                const cellModel = notebook.model.cells.get(i);
                const cell = notebook.widgets[i];
                if (!cell) {
                    continue;
                }
                if (this._isPromptCellModel(cellModel)) {
                    if (!cell.hasClass(PROMPT_CELL_CLASS)) {
                        cell.addClass(PROMPT_CELL_CLASS);
                    }
                }
                if (this._isOutputCellModel(cellModel)) {
                    if (!cell.hasClass(PROMPT_OUTPUT_CLASS)) {
                        cell.addClass(PROMPT_OUTPUT_CLASS);
                    }
                    // Restore "Convert to Cells" button if settings allow
                    if (this._settings?.showConvertButton !== false) {
                        const content = cellModel.getMetadata('jupyvibe_content');
                        if (content && cell.model.type === 'markdown') {
                            this._addConvertButton(panel, cell, content);
                        }
                    }
                }
            }
        };
        // Initial styling
        stylePromptCells();
        // Re-style when cells scroll into view (for windowing mode)
        const onActiveCellChanged = () => {
            stylePromptCells();
        };
        notebook.activeCellChanged.connect(onActiveCellChanged);
        // Listen for cell changes to style new prompt cells
        const cells = notebook.model?.cells;
        const onCellsChanged = () => {
            // Defer to allow widgets to be created
            requestAnimationFrame(() => stylePromptCells());
        };
        if (cells) {
            cells.changed.connect(onCellsChanged);
        }
        // Clean up when notebook is closed
        panel.disposed.connect(() => {
            this._connectors.delete(notebookId);
            notebook.activeCellChanged.disconnect(onActiveCellChanged);
            if (cells) {
                cells.changed.disconnect(onCellsChanged);
            }
        });
    }
    /**
     * Insert a new prompt cell below the active cell.
     */
    insertPromptCell(panel) {
        const notebook = panel.content;
        // Insert a markdown cell below
        _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.NotebookActions.insertBelow(notebook);
        const activeIndex = notebook.activeCellIndex;
        const cell = notebook.widgets[activeIndex];
        const model = cell.model;
        // Mark as prompt cell (no model stored - always use current settings)
        model.setMetadata(PROMPT_METADATA_KEY, {
            isPromptCell: true
        });
        // Change to markdown type for the prompt
        if (notebook.model) {
            const cellData = model.toJSON();
            cellData.cell_type = 'markdown';
            cellData.source = '**AI Prompt:** ';
            notebook.model.sharedModel.deleteCell(activeIndex);
            notebook.model.sharedModel.insertCell(activeIndex, cellData);
        }
        // Add styling class
        const newCell = notebook.widgets[activeIndex];
        newCell.addClass(PROMPT_CELL_CLASS);
        // Focus the cell for editing
        notebook.activeCellIndex = activeIndex;
        notebook.mode = 'edit';
    }
    /**
     * Execute the current prompt cell.
     */
    async executePromptCell(panel) {
        const notebook = panel.content;
        const activeCell = notebook.activeCell;
        if (!activeCell || !this._isPromptCellModel(activeCell.model)) {
            console.log('Not a prompt cell');
            return;
        }
        const connector = this._connectors.get(panel.id);
        if (!connector) {
            console.error('No kernel connector found');
            return;
        }
        // Get model from cell metadata or settings
        const metadata = activeCell.model.getMetadata(PROMPT_METADATA_KEY);
        const defaultModel = this._settings?.defaultModel ?? 'claude-sonnet-4-20250514';
        const model = metadata?.model || defaultModel;
        // Get kernel ID for tool execution
        const kernelId = panel.sessionContext.session?.kernel?.id;
        // Get the prompt text
        const promptText = activeCell.model.sharedModel.getSource();
        // Remove the "**AI Prompt:** " prefix if present
        const cleanPrompt = promptText.replace(/^\*\*AI Prompt:\*\*\s*/i, '');
        // Parse for variable and function references
        const parsed = (0,_promptParser__WEBPACK_IMPORTED_MODULE_2__.parsePrompt)(cleanPrompt);
        // Gather context
        const context = await this._gatherContext(panel, connector, parsed);
        // Process the prompt (substitute variables)
        const variableValues = {};
        for (const [name, info] of Object.entries(context.variables)) {
            variableValues[name] = info.repr;
        }
        const processedPrompt = (0,_promptParser__WEBPACK_IMPORTED_MODULE_2__.processPrompt)(cleanPrompt, variableValues);
        // Insert output cell
        const outputCell = this._insertOutputCell(panel, activeCell);
        // Call the AI backend
        await this._callAI(panel, processedPrompt, context, outputCell, model, kernelId);
    }
    /**
     * Gather context for the prompt including preceding code and referenced items.
     */
    async _gatherContext(panel, connector, parsed) {
        const notebook = panel.content;
        const model = notebook.model;
        const activeIndex = notebook.activeCellIndex;
        // Get preceding code cells and extract images/chart specs from outputs
        const precedingCode = [];
        const images = [];
        const chartSpecs = [];
        // Iterate over the model (not widgets) for robustness under windowing
        if (model) {
            for (let i = 0; i < activeIndex; i++) {
                const cellModel = model.cells.get(i);
                if (!cellModel) {
                    continue;
                }
                if (cellModel.type === 'code') {
                    precedingCode.push(cellModel.sharedModel.getSource());
                    // Extract images and chart specs from code cell outputs
                    if ((0,_jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1__.isCodeCellModel)(cellModel)) {
                        this._extractImagesFromCodeCell(cellModel, i, images);
                        this._extractChartSpecsFromCodeCell(cellModel, i, chartSpecs);
                    }
                }
                else if (cellModel.type === 'markdown') {
                    // Extract images from markdown cell attachments
                    this._extractImagesFromMarkdownCell(cellModel, i, images);
                }
            }
        }
        // Get referenced variables
        const variables = {};
        for (const varName of parsed.variables) {
            const info = await connector.getVariable(varName);
            if (info) {
                variables[varName] = info;
            }
        }
        // Get referenced functions
        const functions = {};
        for (const funcName of parsed.functions) {
            const info = await connector.getFunction(funcName);
            if (info) {
                functions[funcName] = info;
            }
        }
        const conversationHistory = this._gatherConversationHistory(panel, activeIndex);
        return {
            preceding_code: precedingCode.join('\n\n'),
            variables,
            functions,
            images: images.length > 0 ? images : undefined,
            chartSpecs: chartSpecs.length > 0 ? chartSpecs : undefined,
            conversationHistory: conversationHistory.length > 0 ? conversationHistory : undefined
        };
    }
    /**
     * Gather conversation history from previous prompt/response cell pairs.
     * Looks for cells with PROMPT_CELL_CLASS followed by PROMPT_OUTPUT_CLASS.
     */
    _gatherConversationHistory(panel, activeIndex) {
        const notebook = panel.content;
        const model = notebook.model;
        const history = [];
        if (!model) {
            return history;
        }
        let i = 0;
        while (i < activeIndex) {
            const cellModel = model.cells.get(i);
            if (!cellModel) {
                i++;
                continue;
            }
            const cellWidget = notebook.widgets[i];
            if (!cellWidget) {
                i++;
                continue;
            }
            if (cellWidget.hasClass(PROMPT_CELL_CLASS)) {
                const promptText = cellModel.sharedModel.getSource();
                const nextIndex = i + 1;
                if (nextIndex < activeIndex) {
                    const nextWidget = notebook.widgets[nextIndex];
                    const nextModel = model.cells.get(nextIndex);
                    if (nextWidget &&
                        nextModel &&
                        nextWidget.hasClass(PROMPT_OUTPUT_CLASS)) {
                        const responseText = nextModel.sharedModel.getSource();
                        history.push({
                            prompt: promptText,
                            response: responseText
                        });
                        i = nextIndex + 1;
                        continue;
                    }
                }
            }
            i++;
        }
        return history;
    }
    /**
     * Extract images from code cell outputs.
     */
    _extractImagesFromCodeCell(cellModel, cellIndex, images) {
        const outputs = cellModel.outputs;
        if (!outputs) {
            return;
        }
        for (let j = 0; j < outputs.length; j++) {
            const outputModel = outputs.get(j);
            const data = outputModel.data;
            // Check each supported image MIME type
            for (const mimeType of IMAGE_MIME_TYPES) {
                const imageData = data[mimeType];
                if (imageData && typeof imageData === 'string') {
                    images.push({
                        data: imageData,
                        mimeType: mimeType,
                        source: 'output',
                        cellIndex
                    });
                    break; // Only take the first matching image type per output
                }
            }
        }
    }
    /**
     * Extract images from markdown cell attachments.
     */
    _extractImagesFromMarkdownCell(cellModel, cellIndex, images) {
        // Attachments are stored in cell metadata under 'attachments'
        const attachments = cellModel.getMetadata('attachments');
        if (!attachments) {
            return;
        }
        // Iterate through each attachment
        for (const [_filename, mimeData] of Object.entries(attachments)) {
            if (!mimeData || typeof mimeData !== 'object') {
                continue;
            }
            // Check each supported image MIME type
            for (const mimeType of IMAGE_MIME_TYPES) {
                const imageData = mimeData[mimeType];
                if (imageData && typeof imageData === 'string') {
                    images.push({
                        data: imageData,
                        mimeType: mimeType,
                        source: 'attachment',
                        cellIndex
                    });
                    break; // Only take the first matching image type per attachment
                }
            }
        }
    }
    /**
     * Extract chart specs (Vega-Lite, Plotly) from code cell outputs.
     */
    _extractChartSpecsFromCodeCell(cellModel, cellIndex, chartSpecs) {
        const outputs = cellModel.outputs;
        if (!outputs) {
            return;
        }
        for (let j = 0; j < outputs.length; j++) {
            const outputModel = outputs.get(j);
            const data = outputModel.data;
            // Check for Vega-Lite specs (Altair outputs)
            for (const mimeType of Object.keys(data)) {
                if (VEGALITE_MIME_PATTERN.test(mimeType)) {
                    const specData = data[mimeType];
                    if (specData && typeof specData === 'object') {
                        chartSpecs.push({
                            type: 'vega-lite',
                            spec: specData,
                            cellIndex
                        });
                    }
                    break;
                }
            }
            // Check for Plotly specs
            const plotlyData = data[PLOTLY_MIME];
            if (plotlyData && typeof plotlyData === 'object') {
                chartSpecs.push({
                    type: 'plotly',
                    spec: plotlyData,
                    cellIndex
                });
            }
        }
    }
    /**
     * Insert a markdown cell for the AI output.
     * Always creates a new cell for each execution.
     */
    _insertOutputCell(panel, promptCell) {
        const notebook = panel.content;
        const promptIndex = notebook.widgets.indexOf(promptCell);
        // Find where to insert - after the prompt cell and any existing output cells
        let insertAfterIndex = promptIndex;
        for (let i = promptIndex + 1; i < notebook.widgets.length; i++) {
            if (notebook.widgets[i].hasClass(PROMPT_OUTPUT_CLASS)) {
                insertAfterIndex = i;
            }
            else {
                break;
            }
        }
        // Insert new markdown cell after the last output (or after prompt if none)
        notebook.activeCellIndex = insertAfterIndex;
        _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_0__.NotebookActions.insertBelow(notebook);
        const outputIndex = insertAfterIndex + 1;
        const outputCell = notebook.widgets[outputIndex];
        // Set up as output cell
        if (notebook.model) {
            const cellData = outputCell.model.toJSON();
            cellData.cell_type = 'markdown';
            cellData.source = '<div class="jupyvibe-loading">Generating response...</div>';
            notebook.model.sharedModel.deleteCell(outputIndex);
            notebook.model.sharedModel.insertCell(outputIndex, cellData);
        }
        const newOutputCell = notebook.widgets[outputIndex];
        newOutputCell.addClass(PROMPT_OUTPUT_CLASS);
        // Mark as output cell in metadata for persistence across reload
        newOutputCell.model.setMetadata(PROMPT_METADATA_KEY, { isOutputCell: true });
        return newOutputCell;
    }
    /**
     * Call the AI backend and stream the response using signal-based PromptModel.
     */
    async _callAI(panel, prompt, context, outputCell, model, kernelId) {
        // Create or get a PromptModel for this execution
        const promptModel = new _promptModel__WEBPACK_IMPORTED_MODULE_3__.PromptModel();
        // Connect output changes to cell updates
        const onOutputChanged = (_, output) => {
            if (!outputCell.isDisposed) {
                outputCell.model.sharedModel.setSource(output);
            }
        };
        promptModel.outputChanged.connect(onOutputChanged);
        // Abort on cell disposal
        const abortOnDispose = () => promptModel.abort();
        outputCell.disposed.connect(abortOnDispose);
        try {
            const maxSteps = this._settings?.maxToolSteps ?? 5;
            await promptModel.executePrompt(prompt, context, {
                model,
                kernelId,
                maxSteps
            });
            // Render markdown and add convert button
            if (!outputCell.isDisposed && outputCell instanceof _jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1__.MarkdownCell) {
                outputCell.rendered = true;
                const showButton = this._settings?.showConvertButton ?? true;
                if (showButton) {
                    this._addConvertButton(panel, outputCell, promptModel.output);
                }
            }
        }
        catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                return;
            }
            if (!outputCell.isDisposed) {
                outputCell.model.sharedModel.setSource(`**Error:** Failed to connect to AI backend.\n\n${String(error)}`);
                if (outputCell instanceof _jupyterlab_cells__WEBPACK_IMPORTED_MODULE_1__.MarkdownCell) {
                    outputCell.rendered = true;
                }
            }
        }
        finally {
            promptModel.outputChanged.disconnect(onOutputChanged);
            outputCell.disposed.disconnect(abortOnDispose);
            promptModel.dispose();
        }
    }
    /**
     * Check if a cell is a prompt cell.
     */
    isPromptCell(cell) {
        return this._isPromptCellModel(cell.model);
    }
    /**
     * Check if a cell model is a prompt cell.
     */
    _isPromptCellModel(model) {
        const metadata = model.getMetadata(PROMPT_METADATA_KEY);
        return metadata?.isPromptCell === true;
    }
    /**
     * Check if a cell model is an AI output cell.
     */
    _isOutputCellModel(model) {
        const metadata = model.getMetadata(PROMPT_METADATA_KEY);
        return metadata?.isOutputCell === true;
    }
    /**
     * Add a "Convert to Cells" button to an AI response cell.
     * Stores content in cell metadata and adds a persistent button.
     */
    _addConvertButton(panel, cell, content) {
        // Store content in metadata for later retrieval
        cell.model.setMetadata('jupyvibe_content', content);
        // Check if button already exists
        const existingContainer = cell.node.querySelector('.jupyvibe-convert-button-container');
        if (existingContainer) {
            return;
        }
        // Create button container - append directly to cell node
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'jupyvibe-convert-button-container';
        const button = document.createElement('button');
        button.className = 'jp-mod-styled jupyvibe-convert-button';
        button.innerHTML = '<span class="jp-ToolbarButtonComponent-icon"></span><span class="jp-ToolbarButtonComponent-label">Convert to Cells</span>';
        button.title = 'Convert this response into separate code and markdown cells';
        button.addEventListener('click', () => {
            const storedContent = cell.model.getMetadata('jupyvibe_content') || content;
            this._convertToCells(panel, cell, storedContent);
        });
        buttonContainer.appendChild(button);
        // Append directly to cell node (most stable location)
        cell.node.appendChild(buttonContainer);
    }
    /**
     * Convert an AI response cell into native code and markdown cells.
     */
    _convertToCells(panel, responseCell, content) {
        const notebook = panel.content;
        const cellIndex = notebook.widgets.indexOf(responseCell);
        if (cellIndex < 0 || !notebook.model) {
            console.log('[jupyvibe] Convert: invalid cell index or no model');
            return;
        }
        console.log('[jupyvibe] Converting content:', content.substring(0, 200) + '...');
        // Parse the content into blocks
        const blocks = this._parseContentBlocks(content);
        console.log('[jupyvibe] Parsed blocks:', blocks.length, blocks.map(b => ({ type: b.type, len: b.content.length })));
        if (blocks.length === 0) {
            console.log('[jupyvibe] No blocks parsed, keeping original cell');
            return;
        }
        // Remove the response cell
        notebook.model.sharedModel.deleteCell(cellIndex);
        // Insert new cells in reverse order (so they end up in correct order)
        for (let i = blocks.length - 1; i >= 0; i--) {
            const block = blocks[i];
            const cellData = {
                cell_type: block.type === 'code' ? 'code' : 'markdown',
                source: block.content,
                metadata: {}
            };
            notebook.model.sharedModel.insertCell(cellIndex, cellData);
        }
        console.log('[jupyvibe] Inserted', blocks.length, 'cells');
    }
    /**
     * Parse markdown content into code and text blocks.
     */
    _parseContentBlocks(content) {
        const blocks = [];
        // Normalize line endings
        const normalizedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        // Regex to match fenced code blocks - handles:
        // - Optional language specifier
        // - Optional whitespace after language
        // - Code content (non-greedy)
        // - Closing ``` (may be preceded by newline or not)
        const codeBlockRegex = /```(\w*)[ \t]*\n?([\s\S]*?)\n?```/g;
        let lastIndex = 0;
        let match;
        console.log('[jupyvibe] Parsing content, length:', normalizedContent.length);
        console.log('[jupyvibe] Content starts with:', JSON.stringify(normalizedContent.substring(0, 100)));
        while ((match = codeBlockRegex.exec(normalizedContent)) !== null) {
            console.log('[jupyvibe] Found code block match at', match.index, 'language:', match[1], 'code length:', match[2].length);
            // Add any text before this code block
            const textBefore = normalizedContent.slice(lastIndex, match.index).trim();
            if (textBefore) {
                blocks.push({ type: 'markdown', content: textBefore });
            }
            // Add the code block
            const language = match[1] || 'python';
            const code = match[2].trim();
            if (code) {
                blocks.push({ type: 'code', content: code, language });
            }
            lastIndex = match.index + match[0].length;
        }
        // Add any remaining text after the last code block
        const remainingText = normalizedContent.slice(lastIndex).trim();
        if (remainingText) {
            blocks.push({ type: 'markdown', content: remainingText });
        }
        // If no code blocks found but content exists, return as single markdown block
        if (blocks.length === 0 && normalizedContent.trim()) {
            console.log('[jupyvibe] No code blocks found, returning as single markdown');
            blocks.push({ type: 'markdown', content: normalizedContent.trim() });
        }
        return blocks;
    }
}


/***/ },

/***/ "./lib/promptModel.js"
/*!****************************!*\
  !*** ./lib/promptModel.js ***!
  \****************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   PromptModel: () => (/* binding */ PromptModel),
/* harmony export */   PromptModelFactory: () => (/* binding */ PromptModelFactory)
/* harmony export */ });
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/signaling */ "webpack/sharing/consume/default/@lumino/signaling");
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_signaling__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/coreutils */ "webpack/sharing/consume/default/@jupyterlab/coreutils");
/* harmony import */ var _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _toolResultRenderer__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./toolResultRenderer */ "./lib/toolResultRenderer.js");
/**
 * Signal-based prompt model for streaming AI responses.
 *
 * Uses Lumino Signals for reactive updates, following jupyter-ai patterns.
 */



/**
 * Implementation of IPromptModel with signal-based streaming.
 */
class PromptModel {
    constructor() {
        this._state = 'idle';
        this._output = '';
        this._abortController = null;
        this._streamEvent = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
        this._outputChanged = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
        this._stateChanged = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
    }
    /**
     * Signal emitted when streaming events occur.
     */
    get streamEvent() {
        return this._streamEvent;
    }
    /**
     * Signal emitted when the accumulated output changes.
     */
    get outputChanged() {
        return this._outputChanged;
    }
    /**
     * Signal emitted when execution state changes.
     */
    get stateChanged() {
        return this._stateChanged;
    }
    /**
     * Current execution state.
     */
    get state() {
        return this._state;
    }
    /**
     * Current accumulated output text.
     */
    get output() {
        return this._output;
    }
    /**
     * Execute a prompt and stream the response.
     */
    async executePrompt(prompt, context, options) {
        this._setState('executing');
        this._output = '';
        this._abortController = new AbortController();
        this._emitEvent({ type: 'start' });
        const baseUrl = _jupyterlab_coreutils__WEBPACK_IMPORTED_MODULE_1__.PageConfig.getBaseUrl();
        const body = {
            prompt,
            context,
            model: options.model,
            kernel_id: options.kernelId,
            max_steps: options.maxSteps ?? 1
        };
        try {
            const xsrfToken = document.cookie
                .split('; ')
                .find(row => row.startsWith('_xsrf='))
                ?.split('=')[1];
            const response = await fetch(`${baseUrl}jupyvibe/prompt`, {
                method: 'POST',
                credentials: 'same-origin',
                headers: {
                    'Content-Type': 'application/json',
                    ...(xsrfToken && { 'X-XSRFToken': xsrfToken })
                },
                body: JSON.stringify(body),
                signal: this._abortController.signal
            });
            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                try {
                    const errorBody = await response.json();
                    if (errorBody.error) {
                        errorMessage = errorBody.error;
                    }
                }
                catch {
                    // Response wasn't JSON
                }
                throw new Error(errorMessage);
            }
            this._setState('streaming');
            await this._processStream(response);
            this._setState('idle');
        }
        catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                this._setState('idle');
                return;
            }
            this._emitEvent({
                type: 'error',
                error: error instanceof Error ? error.message : String(error)
            });
            this._setState('error');
            throw error;
        }
        finally {
            this._abortController = null;
        }
    }
    /**
     * Process the SSE stream from the server.
     */
    async _processStream(response) {
        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('No response body');
        }
        const decoder = new TextDecoder();
        let buffer = '';
        let currentToolCall = null;
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done)
                    break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const rawLine of lines) {
                    const line = rawLine.replace(/\r$/, '');
                    if (!line.startsWith('data: '))
                        continue;
                    const data = line.slice(6);
                    try {
                        const event = JSON.parse(data);
                        this._handleServerEvent(event, currentToolCall);
                        // Track tool call state
                        if (event.tool_call) {
                            currentToolCall = {
                                name: event.tool_call.name,
                                id: event.tool_call.id,
                                input: ''
                            };
                        }
                        else if (event.tool_input && currentToolCall) {
                            currentToolCall.input += event.tool_input;
                        }
                        else if (event.tool_result) {
                            currentToolCall = null;
                        }
                    }
                    catch {
                        // Ignore invalid JSON
                    }
                }
            }
        }
        finally {
            reader.releaseLock();
        }
        this._emitEvent({ type: 'done' });
    }
    /**
     * Handle a server-sent event.
     */
    _handleServerEvent(event, currentToolCall) {
        if (event.text) {
            const text = event.text;
            this._appendOutput(text);
            this._emitEvent({ type: 'text', text });
        }
        else if (event.error) {
            const error = event.error;
            this._appendOutput(`\n\n**Error:** ${error}\n`);
            this._emitEvent({ type: 'error', error });
        }
        else if (event.tool_call) {
            const toolCall = event.tool_call;
            this._appendOutput(`\n\n🔧 *Calling tool: \`${toolCall.name}\`...*\n`);
            this._emitEvent({
                type: 'tool_call',
                toolCall: { name: toolCall.name, id: toolCall.id }
            });
        }
        else if (event.tool_input) {
            this._emitEvent({ type: 'tool_input', toolInput: event.tool_input });
        }
        else if (event.tool_result) {
            const tr = event.tool_result;
            const rendered = (0,_toolResultRenderer__WEBPACK_IMPORTED_MODULE_2__.renderToolResult)(tr.result);
            this._appendOutput(rendered);
            this._emitEvent({
                type: 'tool_result',
                toolResult: { id: tr.id, name: tr.name, result: tr.result }
            });
        }
    }
    /**
     * Append text to output and emit change signal.
     */
    _appendOutput(text) {
        this._output += text;
        this._outputChanged.emit(this._output);
    }
    /**
     * Abort the current execution.
     */
    abort() {
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
    }
    /**
     * Reset the model state.
     */
    reset() {
        this.abort();
        this._output = '';
        this._setState('idle');
        this._outputChanged.emit(this._output);
    }
    /**
     * Set state and emit signal.
     */
    _setState(state) {
        if (this._state !== state) {
            this._state = state;
            this._stateChanged.emit(state);
        }
    }
    /**
     * Emit a stream event.
     */
    _emitEvent(event) {
        this._streamEvent.emit(event);
    }
    /**
     * Dispose of the model.
     */
    dispose() {
        this.abort();
        _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal.clearData(this);
    }
}
/**
 * Factory for creating PromptModel instances.
 */
class PromptModelFactory {
    create() {
        return new PromptModel();
    }
}


/***/ },

/***/ "./lib/promptParser.js"
/*!*****************************!*\
  !*** ./lib/promptParser.js ***!
  \*****************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   parsePrompt: () => (/* binding */ parsePrompt),
/* harmony export */   processPrompt: () => (/* binding */ processPrompt),
/* harmony export */   removeFunctionReferences: () => (/* binding */ removeFunctionReferences),
/* harmony export */   substituteVariables: () => (/* binding */ substituteVariables)
/* harmony export */ });
/**
 * Parser for $`variable` and &`function` syntax in prompts.
 */
/**
 * Parse a prompt to extract variable and function references.
 *
 * - $`variableName` references a kernel variable
 * - &`functionName` makes a function available as an AI tool
 */
function parsePrompt(text) {
    // Match $`variableName` (word characters inside backticks after $)
    const variablePattern = /\$`([a-zA-Z_][a-zA-Z0-9_]*)`/g;
    // Match &`functionName` (word characters inside backticks after &)
    const functionPattern = /&`([a-zA-Z_][a-zA-Z0-9_]*)`/g;
    const variables = [];
    const functions = [];
    let match;
    // Find all variable references
    while ((match = variablePattern.exec(text)) !== null) {
        const varName = match[1];
        if (!variables.includes(varName)) {
            variables.push(varName);
        }
    }
    // Find all function references
    while ((match = functionPattern.exec(text)) !== null) {
        const funcName = match[1];
        if (!functions.includes(funcName)) {
            functions.push(funcName);
        }
    }
    return {
        variables,
        functions
    };
}
/**
 * Replace variable references in prompt with their values.
 * Uses a replacer function to safely handle $ and other special chars in values.
 */
function substituteVariables(text, variableValues) {
    let result = text;
    for (const [name, value] of Object.entries(variableValues)) {
        const pattern = new RegExp(`\\$\`${name}\``, 'g');
        // Use replacer function to avoid interpreting $& etc. in value
        result = result.replace(pattern, () => value);
    }
    return result;
}
/**
 * Remove function references from the prompt text.
 * (They're used for tool definitions, not prompt content)
 * Also normalizes whitespace to avoid double spaces.
 */
function removeFunctionReferences(text) {
    return text
        .replace(/&`([a-zA-Z_][a-zA-Z0-9_]*)`/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}
/**
 * Get a cleaned prompt with variables substituted and function refs removed.
 */
function processPrompt(text, variableValues) {
    let result = substituteVariables(text, variableValues);
    result = removeFunctionReferences(result);
    return result.trim();
}


/***/ },

/***/ "./lib/settings.js"
/*!*************************!*\
  !*** ./lib/settings.js ***!
  \*************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   SettingsManager: () => (/* binding */ SettingsManager)
/* harmony export */ });
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/signaling */ "webpack/sharing/consume/default/@lumino/signaling");
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_signaling__WEBPACK_IMPORTED_MODULE_0__);
/**
 * Settings management for jupyvibe extension.
 *
 * Integrates with JupyterLab's ISettingRegistry for persistent configuration.
 */

const PLUGIN_ID = 'jupyvibe:plugin';
/**
 * Default settings values.
 */
const DEFAULT_SETTINGS = {
    provider: 'anthropic',
    defaultModel: 'claude-sonnet-4-20250514',
    maxToolSteps: 5,
    showConvertButton: true
};
/**
 * Settings manager that wraps ISettingRegistry.
 */
class SettingsManager {
    constructor() {
        this._settings = null;
        this._provider = DEFAULT_SETTINGS.provider;
        this._defaultModel = DEFAULT_SETTINGS.defaultModel;
        this._maxToolSteps = DEFAULT_SETTINGS.maxToolSteps;
        this._showConvertButton = DEFAULT_SETTINGS.showConvertButton;
        this._settingsChanged = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
    }
    /**
     * Signal emitted when settings change.
     */
    get settingsChanged() {
        return this._settingsChanged;
    }
    get provider() {
        return this._provider;
    }
    get defaultModel() {
        return this._defaultModel;
    }
    get maxToolSteps() {
        return this._maxToolSteps;
    }
    get showConvertButton() {
        return this._showConvertButton;
    }
    /**
     * Initialize settings from the registry.
     */
    async initialize(registry) {
        try {
            this._settings = await registry.load(PLUGIN_ID);
            this._updateFromSettings();
            this._settings.changed.connect(this._onSettingsChanged, this);
        }
        catch (error) {
            console.warn('[jupyvibe] Failed to load settings, using defaults:', error);
        }
    }
    /**
     * Update a setting value.
     */
    async set(key, value) {
        if (this._settings) {
            await this._settings.set(key, value);
        }
    }
    /**
     * Get all settings as a plain object.
     */
    toJSON() {
        return {
            provider: this._provider,
            defaultModel: this._defaultModel,
            maxToolSteps: this._maxToolSteps,
            showConvertButton: this._showConvertButton
        };
    }
    _onSettingsChanged() {
        this._updateFromSettings();
        this._settingsChanged.emit();
    }
    _updateFromSettings() {
        if (!this._settings)
            return;
        const composite = this._settings.composite;
        this._provider =
            composite['provider'] ?? DEFAULT_SETTINGS.provider;
        this._defaultModel =
            composite['defaultModel'] ?? DEFAULT_SETTINGS.defaultModel;
        this._maxToolSteps =
            composite['maxToolSteps'] ?? DEFAULT_SETTINGS.maxToolSteps;
        this._showConvertButton =
            composite['showConvertButton'] ?? DEFAULT_SETTINGS.showConvertButton;
    }
    /**
     * Dispose of the settings manager.
     */
    dispose() {
        if (this._settings) {
            this._settings.changed.disconnect(this._onSettingsChanged, this);
        }
        _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal.clearData(this);
    }
}


/***/ },

/***/ "./lib/tokens.js"
/*!***********************!*\
  !*** ./lib/tokens.js ***!
  \***********************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   IExtensionSettings: () => (/* binding */ IExtensionSettings),
/* harmony export */   IKernelConnectorFactory: () => (/* binding */ IKernelConnectorFactory),
/* harmony export */   IPromptCellManager: () => (/* binding */ IPromptCellManager),
/* harmony export */   IPromptModelFactory: () => (/* binding */ IPromptModelFactory)
/* harmony export */ });
/* harmony import */ var _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/coreutils */ "webpack/sharing/consume/default/@lumino/coreutils");
/* harmony import */ var _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__);
/**
 * Dependency injection tokens for jupyvibe extension.
 *
 * Following JupyterLab's plugin architecture pattern, these tokens
 * allow loose coupling between components and enable testing/mocking.
 */

/**
 * Token for the kernel connector factory.
 */
const IKernelConnectorFactory = new _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__.Token('jupyvibe:IKernelConnectorFactory', 'Factory for creating kernel connectors');
/**
 * Token for the prompt cell manager.
 */
const IPromptCellManager = new _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__.Token('jupyvibe:IPromptCellManager', 'Manages prompt cells within notebooks');
/**
 * Token for the prompt model factory.
 */
const IPromptModelFactory = new _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__.Token('jupyvibe:IPromptModelFactory', 'Factory for creating prompt models');
/**
 * Token for extension settings.
 */
const IExtensionSettings = new _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__.Token('jupyvibe:IExtensionSettings', 'Extension configuration settings');


/***/ },

/***/ "./lib/toolResultRenderer.js"
/*!***********************************!*\
  !*** ./lib/toolResultRenderer.js ***!
  \***********************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   renderToolResult: () => (/* binding */ renderToolResult)
/* harmony export */ });
/**
 * Utility functions for rendering tool results as markdown.
 */
/**
 * Render a structured tool result into markdown.
 */
function renderToolResult(result) {
    if (!result || typeof result !== 'object') {
        return `\n**Tool Result:** ${JSON.stringify(result)}\n`;
    }
    const resultObj = result;
    // Handle error status
    if (resultObj.status === 'error' || resultObj.error) {
        return `\n**Tool Error:** ${resultObj.error || 'Unknown error'}\n`;
    }
    const type = resultObj.type;
    const content = resultObj.content ?? '';
    if (type === 'text') {
        return `\n**Tool Result:**\n\`\`\`\n${content}\n\`\`\`\n`;
    }
    if (type === 'html') {
        // Raw HTML is allowed in Jupyter markdown
        return `\n**Tool Result (HTML):**\n\n${content}\n`;
    }
    if (type === 'image') {
        const format = resultObj.format || 'png';
        return `\n**Tool Result:**\n\n![](data:image/${format};base64,${content})\n`;
    }
    // Fallback
    return `\n**Tool Result:**\n\`\`\`json\n${JSON.stringify(result, null, 2)}\n\`\`\`\n`;
}


/***/ }

}]);
//# sourceMappingURL=lib_index_js.0cd05c16f889cdf79cd5.js.map